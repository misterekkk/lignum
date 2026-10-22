#include <lignum/model.hpp>

#include <gtest/gtest.h>

#include <cstddef>

TEST(Layout, KNodeFillsExactlyOneCacheLine) {
    EXPECT_EQ(sizeof(lignum::KNode), 64u);
    EXPECT_EQ(alignof(lignum::KNode), 64u);
}

TEST(Layout, KNodeSlotsAreNaturallyAligned) {
    EXPECT_EQ(offsetof(lignum::KNode, thresholds) % alignof(double), 0u);
    EXPECT_EQ(offsetof(lignum::KNode, children) % alignof(int32_t), 0u);
}
