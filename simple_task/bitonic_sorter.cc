// Bitonic sorter
// The algorithm is described here https://en.wikipedia.org/wiki/Bitonic_sorter
// Author: dongyan (Andy)

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include "legion.h"

#if DEBUG == 1
    #define debug(...) fprintf(stderr, __VA_ARGS__)
#else
    #define debug(...) 0
#endif

using namespace Legion;

enum {
    TOP_LEVEL_TASK_ID,
    SUBSORTER_TASK_ID,
    SINGLE_SWAP_TASK_ID,
};

template<typename T>
struct MyVec {
    std::vector<T> vec;

    MyVec(size_t sz = 0): vec(sz) {}
    MyVec(std::initializer_list<T> l): vec(l) {}
    size_t size() const { return vec.size(); }
    T& operator[](int i) { return vec[i]; }
    const T& operator[](int i) const { return vec[i]; }
    void append(const T& e) { vec.push_back(e); }

    size_t legion_buffer_size(void) const {
        size_t result = sizeof(size_t);
        for (const auto &e : vec) {
            result += sizeof(e);
        }
        debug("buffer size: %zu", result);
        return result;
    }

    size_t legion_serialize(void *buffer) const {
        char *target = (char *)buffer;
        *(size_t *)target = vec.size();
        target += sizeof(size_t);
        for (const auto &e : vec) {
            *(T*)target = e;
            target += sizeof(e);
        }
        debug("finish serializing");
        return (size_t)target - (size_t)buffer;
    }

    size_t legion_deserialize(const void *buffer) {
        const char *source = (const char *)buffer;
        size_t length = *(const size_t *)source;
        source += sizeof(size_t);
        vec.resize(length);
        for (auto &e : vec) {
            e = *(const T*)source;
            source += sizeof(e);
        }
        debug("finish deserializing");
        return (size_t)source - (size_t)buffer;
    }
};

void print_myvec(const MyVec<int> &sorted, int start, int end) {
    for (int i = start; i < end; i++) {
        if (sorted[i] == INT_MAX) {
            printf("# ");
        } else {
            printf("%d ", sorted[i]);
        }
    }
    printf("\n");
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    int num_inputs = 0;
    std::vector<int> nums;

    // handle inputs
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (command_args.argv[i][0] == '-') {
            i++;
            continue;
        }
        int input = atoi(command_args.argv[i]);
        nums.push_back(input);
        num_inputs++;
    }
    assert(num_inputs > 0);

    // find the next-least power of 2,
    // and to fill up with max values
    int num_total = num_inputs;
    while (num_total != (num_total & (-num_total))) {
        num_total += (num_total & (-num_total));
    }
    for (int i = num_inputs; i < num_total; i++) {
        nums.push_back(INT_MAX);
    }

    printf("Running bitonic sorter for %d inputs...\n", num_inputs);

    // First, do single swaps to acquire initial future results
    std::vector<std::vector<Future>> iterResults;
    std::vector<Future> results;
    for (int lo = 0; lo < num_total; lo += 2) {
        debug("input: %d %d\n", nums[lo], nums[lo+1]);
        int args[] = {nums[lo], nums[lo+1]};
        TaskLauncher single_swaper(SINGLE_SWAP_TASK_ID, TaskArgument(&args[0], sizeof(int) * 2));
        Future res = runtime->execute_task(ctx, single_swaper);
        results.push_back(res);
    }
    iterResults.push_back(results);

    // Then iteratively merge sorting results from previous operations,
    for (int gap = 4; gap <= num_total; gap <<= 1) {
        int j = 0;
        std::vector<Future> results;
        for (int lo = 0; lo < num_total; lo += gap) {
            // spawn a sub-task for each subsorter block
            // a subsorter requires the sorting results of two previous subsorters
            TaskLauncher subsorter(SUBSORTER_TASK_ID, TaskArgument(NULL, 0));
            subsorter.add_future(iterResults.back()[j * 2]);
            subsorter.add_future(iterResults.back()[j * 2 + 1]);
            Future res = runtime->execute_task(ctx, subsorter);
            results.push_back(res);
            j++;
        }
        iterResults.push_back(results);
    }
    assert(iterResults.back().size() == 1);

    auto final_result = iterResults.back()[0];
    const auto &sorted = final_result.get_result<MyVec<int>>();
    assert(sorted.size() == num_total);

    // print result
    printf("sorting results: ");
    print_myvec(sorted, 0, num_inputs);
}

MyVec<int> subsorter_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
    assert(task->futures.size() == 2);

    Future f1 = task->futures[0];
    auto vec1 = f1.get_result<MyVec<int>>();
    Future f2 = task->futures[1];
    auto vec2 = f2.get_result<MyVec<int>>();

    assert(vec1.size() == vec2.size());
    int num_vec = vec1.size();
    int num_total = num_vec * 2;

    MyVec<int> sorted(num_total);
    std::vector<Future> results;

    // First do crosswork,
    // split the sorted subsequences into bitonic subsequences
    //
    // launch tasks
    for (int i = 0; i < num_vec; i++) {
        int args[] = {vec1[i], vec2[num_vec-i-1]};
        TaskLauncher launcher(SINGLE_SWAP_TASK_ID, TaskArgument(&args[0], sizeof(int) * 2));
        Future res = runtime->execute_task(ctx, launcher);
        results.push_back(res);
    }
    // get results
    for (int i = 0; i < num_vec; i++) {
        auto values = results[i].get_result<MyVec<int>>();
        sorted[i] = values[0];
        sorted[num_total-i-1] = values[1];
    }
    results.clear();

    // Then sort each bitonic subsequence
    for (int gap = num_vec; gap > 1; gap /= 2) {
        // launch tasks
        for (int lo = 0; lo < num_total; lo += gap) {
            int half_sz = gap / 2;
            for (int i = 0; i < half_sz; i++) {
                int args[] = {sorted[lo+i], sorted[lo+i+half_sz]};
                TaskLauncher launcher(SINGLE_SWAP_TASK_ID, TaskArgument(&args[0], sizeof(int) * 2));
                Future res = runtime->execute_task(ctx, launcher);
                results.push_back(res);
            }
        }
        // get results
        int j = 0;
        for (int lo = 0; lo < num_total; lo += gap) {
            int half_sz = gap / 2;
            for (int i = 0; i < half_sz; i++) {
                auto values = results[j].get_result<MyVec<int>>();
                sorted[lo+i] = values[0];
                sorted[lo+i+half_sz] = values[1];
                j++;
            }
        }
        results.clear();
    }

    // may get disordered output ?
    printf("subsorter results: ");
    print_myvec(sorted, 0, num_total);
    return sorted;
}

MyVec<int> single_swap_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(int) * 2);
    auto values = (const int *)(task->args);
    debug("swap: %d %d\n", values[0], values[1]);
    MyVec<int> result {std::min(values[0], values[1]), std::max(values[0], values[1])};
    return result;
}

int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }

    {
        TaskVariantRegistrar registrar(SUBSORTER_TASK_ID, "subsorter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<MyVec<int>, subsorter_task>(registrar, "subsorter");
    }

    {
        TaskVariantRegistrar registrar(SINGLE_SWAP_TASK_ID, "single_swap");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<MyVec<int>, single_swap_task>(registrar, "single_swap");
    }

    return Runtime::start(argc, argv);
}
