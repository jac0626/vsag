// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <new>

#include "datacell/dense_duplicate_tracker.h"
#include "hgraph.h"  // IWYU pragma: keep
#include "impl/pruning_strategy.h"
#include "utils/util_functions.h"

namespace vsag {

uint32_t
HGraph::Remove(const std::vector<int64_t>& ids, RemoveMode mode) {
    uint32_t delete_count = 0;
    if (mode == RemoveMode::MARK_REMOVE) {
        std::scoped_lock label_lock(this->label_lookup_mutex_);
        delete_count = this->label_table_->MarkRemove(ids);
        delete_count_ += delete_count;
        return delete_count;
    }

    if (mode == RemoveMode::FORCE_REMOVE) {
        CHECK_ARGUMENT(this->support_force_remove(),
                       "force remove requires index_param.support_force_remove to be true");
        std::unique_lock<std::shared_mutex> wlock(this->force_remove_mutex_);
        std::unique_lock<std::shared_mutex> dedup_storage_lock;
        if (this->using_dedup_storage()) {
            dedup_storage_lock = std::unique_lock<std::shared_mutex>(this->global_mutex_);
        }
        for (const auto& id : ids) {
            delete_count += this->force_remove_one(id);
        }
        if (delete_count != 0) {
            try {
                this->shrink_to_fit();
            } catch (const VsagException& e) {
                if (e.error_.type != ErrorType::NO_ENOUGH_MEMORY) {
                    throw;
                }
                // Deletion has already completed; compaction is best effort when memory is exhausted.
            } catch (const std::bad_alloc&) {
                // SafeAllocator reports failed storage shrinking as std::bad_alloc.
            }
            if (this->using_dedup_storage()) {
                this->cal_memory_usage();
            }
        }
        return delete_count;
    }

    throw VsagException(ErrorType::INVALID_ARGUMENT, "RemoveMode not supported");
}

void
HGraph::find_new_entry_point() {
    bool find_new_ep = false;
    auto inner_id = this->entry_point_id_;
    while (not route_graphs_.empty()) {
        auto& upper_graph = route_graphs_.back();
        Vector<InnerIdType> neighbors(allocator_);
        upper_graph->GetNeighbors(this->entry_point_id_, neighbors);
        for (const auto& nb_id : neighbors) {
            if (inner_id == nb_id) {
                continue;
            }
            this->entry_point_id_ = nb_id;
            find_new_ep = true;
            break;
        }
        if (find_new_ep) {
            break;
        }
        route_graphs_.pop_back();
    }
}

void
HGraph::graph_force_remove_one(const InnerIdType& inner_id,
                               const FlattenInterfacePtr& flatten,
                               const GraphInterfacePtr& graph) {
    Vector<InnerIdType> forward_neighbors(allocator_);
    graph->GetNeighbors(inner_id, forward_neighbors);
    Vector<InnerIdType> reverse_neighbors(allocator_);
    graph->GetIncomingNeighbors(inner_id, reverse_neighbors);
    if (forward_neighbors.empty() && reverse_neighbors.empty()) {
        return;
    }

    UnorderedSet<InnerIdType> affected_nodes(allocator_);
    auto current_count = this->total_count_.load();
    for (const auto& n : forward_neighbors) {
        if (n < current_count) {
            affected_nodes.insert(n);
        }
    }
    for (const auto& n : reverse_neighbors) {
        if (n < current_count) {
            affected_nodes.insert(n);
        }
    }

    auto max_degree = graph->MaximumDegree();

    for (const auto& neighbor : affected_nodes) {
        LockGuard lock(neighbors_mutex_, neighbor);

        Vector<InnerIdType> neighbors_of_neighbor(allocator_);
        graph->GetNeighbors(neighbor, neighbors_of_neighbor);

        UnorderedSet<InnerIdType> candidate_set(allocator_);
        for (const auto& nb : neighbors_of_neighbor) {
            if (nb != inner_id) {
                candidate_set.insert(nb);
            }
        }
        for (const auto& nb : forward_neighbors) {
            if (nb != inner_id && nb != neighbor) {
                candidate_set.insert(nb);
            }
        }

        Vector<InnerIdType> candidate_list(allocator_);
        auto current_count = this->total_count_.load();
        for (const auto& candidate : candidate_set) {
            if (candidate < current_count) {
                candidate_list.emplace_back(candidate);
            }
        }

        select_edges_by_heuristic(
            candidate_list, neighbor, max_degree, flatten, allocator_, alpha_);

        graph->InsertNeighborsById(neighbor, candidate_list);
    }

    Vector<InnerIdType> empty_neighbor(allocator_);
    graph->InsertNeighborsById(inner_id, empty_neighbor);
}

void
HGraph::move_id(InnerIdType from, InnerIdType to) {
    basic_flatten_codes_->Move(from, to);
    if (high_precise_codes_) {
        high_precise_codes_->Move(from, to);
    }

    if (extra_infos_) {
        extra_infos_->Move(from, to);
    }

    bottom_graph_->Move(from, to);
    for (const auto& route_graph : route_graphs_) {
        route_graph->Move(from, to);
    }

    label_table_->Move(from, to);

    if (entry_point_id_ == from) {
        entry_point_id_ = to;
    }
}

void
HGraph::move_graph_id(InnerIdType from, InnerIdType to) {
    if (from == to) {
        return;
    }

    bottom_graph_->Move(from, to);
    for (const auto& route_graph : route_graphs_) {
        route_graph->Move(from, to);
    }
    if (entry_point_id_ == from) {
        entry_point_id_ = to;
    }
}

void
HGraph::move_logical_metadata(InnerIdType from, InnerIdType to) {
    if (extra_infos_) {
        extra_infos_->Move(from, to);
    }
    label_table_->Move(from, to);
}

void
HGraph::update_graphs_for_deduplicated_remove(InnerIdType inner_id, InnerIdType swap_id) {
    auto tracker = this->bottom_graph_->GetDuplicateTracker();
    if (tracker == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "deduplicated HGraph force remove requires a duplicate tracker");
    }

    const auto removed_representative = tracker->GetGroupId(inner_id);
    const auto swap_representative = tracker->GetGroupId(swap_id);
    const auto removed_duplicates = tracker->GetDuplicateIds(inner_id);
    const bool same_group = removed_representative == swap_representative;
    const bool removed_group_is_singleton = removed_duplicates.empty();
    const bool removed_entry_point = entry_point_id_ == inner_id;

    const auto graph_repair_codes = this->get_precise_codes();
    auto remove_graph_id = [this, &graph_repair_codes](InnerIdType id) {
        if (id == this->entry_point_id_) {
            this->find_new_entry_point();
        }
        this->graph_force_remove_one(id, graph_repair_codes, this->bottom_graph_);
        for (const auto& route_graph : this->route_graphs_) {
            this->graph_force_remove_one(id, graph_repair_codes, route_graph);
        }
    };

    if (same_group) {
        if (removed_group_is_singleton) {
            remove_graph_id(inner_id);
        }
    } else {
        if (removed_representative == inner_id) {
            if (removed_group_is_singleton) {
                remove_graph_id(inner_id);
            } else {
                const auto new_representative =
                    *std::min_element(removed_duplicates.begin(), removed_duplicates.end());
                this->move_graph_id(inner_id, new_representative);
            }
        }

        if (inner_id < swap_representative) {
            this->move_graph_id(swap_representative, inner_id);
        }
    }

    if (removed_group_is_singleton && removed_entry_point && entry_point_id_ == inner_id) {
        if (swap_id != inner_id) {
            entry_point_id_ = std::min(inner_id, swap_representative);
        } else if (inner_id > 0) {
            entry_point_id_ = tracker->GetGroupId(0);
        } else {
            entry_point_id_ = INVALID_ENTRY_POINT;
        }
    }
}

uint32_t
HGraph::force_remove_one(int64_t label) {
    InnerIdType inner_id;
    {
        std::shared_lock lock(this->label_lookup_mutex_);
        bool found = false;
        std::tie(found, inner_id) = this->label_table_->TryGetIdByLabel(label, true);
        if (not found) {
            return 0;
        }
    }
    InnerIdType swap_id = this->total_count_.load() - 1;
    if (this->using_dedup_storage()) {
        this->update_graphs_for_deduplicated_remove(inner_id, swap_id);
    } else {
        if (inner_id == this->entry_point_id_) {
            this->find_new_entry_point();
        }
        graph_force_remove_one(inner_id, basic_flatten_codes_, bottom_graph_);
        for (const auto& route_graph : route_graphs_) {
            graph_force_remove_one(inner_id, basic_flatten_codes_, route_graph);
        }
    }

    bool was_mark_removed = false;
    {
        std::unique_lock lock(this->label_lookup_mutex_);
        was_mark_removed = this->label_table_->IsRemoved(inner_id);
        this->label_table_->ForceRemove(label, inner_id);
        if (swap_id != inner_id) {
            if (this->using_dedup_storage()) {
                this->move_logical_metadata(swap_id, inner_id);
            } else {
                this->move_id(swap_id, inner_id);
            }
        }
        if (this->using_dedup_storage()) {
            auto tracker = std::dynamic_pointer_cast<DenseDuplicateTracker>(
                this->bottom_graph_->GetDuplicateTracker());
            if (tracker == nullptr) {
                throw VsagException(
                    ErrorType::INTERNAL_ERROR,
                    "deduplicated HGraph force remove requires a dense duplicate tracker");
            }
            tracker->RemoveAndSwapLast(inner_id, swap_id);
            this->code_slot_map_->RemoveAndSwapLast(inner_id, swap_id);
        }
    }
    if (was_mark_removed) {
        this->delete_count_.fetch_sub(1);
    }
    this->total_count_.fetch_sub(1);
    return 1;
}

void
HGraph::compact_deduplicated_codes() {
    std::unique_lock<std::shared_mutex> codes_lock(this->persistent_codes_mutex_);
    const auto physical_count =
        this->code_slot_map_->CompactSlots([this](CodeSlotIdType from, CodeSlotIdType to) {
            GetCodeSlotPhysicalFlatten(this->basic_flatten_codes_)->Move(from, to);
            if (this->has_precise_reorder()) {
                GetCodeSlotPhysicalFlatten(this->high_precise_codes_)->Move(from, to);
            }
            if (this->create_new_raw_vector_) {
                GetCodeSlotPhysicalFlatten(this->raw_vector_)->Move(from, to);
            }
        });

    GetCodeSlotPhysicalFlatten(this->basic_flatten_codes_)->SetTotalCount(physical_count);
    if (this->has_precise_reorder()) {
        GetCodeSlotPhysicalFlatten(this->high_precise_codes_)->SetTotalCount(physical_count);
    }
    if (this->create_new_raw_vector_) {
        GetCodeSlotPhysicalFlatten(this->raw_vector_)->SetTotalCount(physical_count);
    }
    this->physical_code_capacity_.store(physical_count, std::memory_order_release);
}

void
HGraph::shrink_to_fit() {
    auto total_count = this->total_count_.load();

    if (this->using_dedup_storage()) {
        this->compact_deduplicated_codes();
        const auto physical_count = this->code_slot_map_->PhysicalCount();
        GetCodeSlotPhysicalFlatten(this->basic_flatten_codes_)->ShrinkToFit(physical_count);
        if (this->has_precise_reorder()) {
            GetCodeSlotPhysicalFlatten(this->high_precise_codes_)->ShrinkToFit(physical_count);
        }
        if (this->create_new_raw_vector_) {
            GetCodeSlotPhysicalFlatten(this->raw_vector_)->ShrinkToFit(physical_count);
        }
    } else {
        basic_flatten_codes_->ShrinkToFit(total_count);
        if (high_precise_codes_) {
            high_precise_codes_->ShrinkToFit(total_count);
        }
    }
    bottom_graph_->ShrinkToFit(total_count);
    for (const auto& route_graph : route_graphs_) {
        route_graph->ShrinkToFit(total_count);
    }
    label_table_->ShrinkToFit(total_count);
}

void
HGraph::UpdateAttribute(int64_t id, const AttributeSet& new_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0);
}

void
HGraph::UpdateAttribute(int64_t id,
                        const AttributeSet& new_attrs,
                        const AttributeSet& origin_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0, origin_attrs);
}

}  // namespace vsag
