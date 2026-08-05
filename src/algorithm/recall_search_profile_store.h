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

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "json_types.h"
#include "vsag/recall_search_profile.h"

namespace vsag {

class RecallSearchProfileStore {
public:
    using SearchParametersPtr = std::shared_ptr<const std::string>;
    using EntryValidator = std::function<void(const RecallSearchProfileEntry&)>;

    RecallSearchProfileStore();

    void
    Update(const RecallSearchProfileEntry& entry);

    [[nodiscard]] SearchParametersPtr
    Resolve(int64_t top_k, double target_recall, const std::string& path) const;

    void
    AppendTo(JsonType& basic_info) const;

    void
    RestoreFrom(const JsonType& basic_info, const EntryValidator& validator = {});

private:
    struct Snapshot;

    mutable std::mutex update_mutex_{};
    std::shared_ptr<const Snapshot> snapshot_{};
};

}  // namespace vsag
