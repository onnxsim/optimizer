// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <onnxoptimizer/optimize.h>
#include <onnx/defs/parser.h>

#include <unordered_set>

#include "onnxoptimizer/passes/cse_util.h"
#include "onnxoptimizer/passes/tensor_content_hash.h"

TEST(OptimizerTest, NopReshape) {
    const char* graph_str = R"(
        <
            ir_version: 7,
            opset_import: [ "": 10]
        >
        agraph (float[5, 7] X) => (float[5, 7] Z)
        {
            Shape = Constant<value=int64[2]{5, -1}> ()
            Y = Reshape (X, Shape)
            Z = Identity(Y)
        }
    )";
    onnx::ModelProto model;
    const auto status = onnx::OnnxParser::Parse(model, graph_str);
    EXPECT_TRUE(status.IsOK());
    auto optimized_model = onnx::optimization::Optimize(model, {"eliminate_nop_reshape", "eliminate_deadend"});

    ASSERT_EQ(optimized_model.graph().node().size(), 1);
    ASSERT_EQ(optimized_model.graph().node()[0].op_type(), "Identity");
}

// Exercises the Graph-native entry points (Optimizer::optimize(Graph&),
// OptimizeGraph/OptimizeGraphFixed) added for onnxsim issue #633: a C++
// caller that already holds a Graph should be able to run passes directly
// on it, without any ModelProto <-> Graph round trip.
TEST(OptimizerTest, OptimizeGraphInPlace) {
    const char* graph_str = R"(
        <
            ir_version: 7,
            opset_import: [ "": 10]
        >
        agraph (float[5, 7] X) => (float[5, 7] Z)
        {
            Shape = Constant<value=int64[2]{5, -1}> ()
            Y = Reshape (X, Shape)
            Z = Identity(Y)
        }
    )";
    onnx::ModelProto model;
    const auto status = onnx::OnnxParser::Parse(model, graph_str);
    EXPECT_TRUE(status.IsOK());

    std::shared_ptr<onnx::Graph> graph(onnx::ImportModelProto(model));
    ASSERT_NE(graph.get(), nullptr);

    std::map<std::string, unsigned int> report;
    onnx::optimization::OptimizeGraph(
        *graph, {"eliminate_nop_reshape", "eliminate_deadend"}, &report);
    ASSERT_GT(report.size(), 0u);

    onnx::ModelProto optimized_model = onnx::PrepareOutput(model);
    onnx::ExportModelProto(&optimized_model, graph);

    ASSERT_EQ(optimized_model.graph().node().size(), 1);
    ASSERT_EQ(optimized_model.graph().node()[0].op_type(), "Identity");
}

// EliminateDuplicateInitializer (cse_util.h's CSETensorHash/CSETensorCompare)
// under both settings of GetTrustTensorContentHash(): the BLAKE3-digest fast
// path (default) and the exact, pre-hashing fallback. Covers a raw_data
// initializer pair and a typed-field (non-raw_data) initializer pair, since
// TensorContentDigest handles those two representations differently.
TEST(OptimizerTest, EliminateDuplicateInitializerTrustHash) {
    for (bool trust : {true, false}) {
        onnx::optimization::SetTrustTensorContentHash(trust);

        onnx::ModelProto model;
        model.set_ir_version(7);
        model.add_opset_import()->set_version(10);
        auto* graph = model.mutable_graph();
        graph->set_name("g");

        auto add_init = [&](const std::string& name,
                             onnx::TensorProto_DataType dtype,
                             const std::vector<int64_t>& dims) -> onnx::TensorProto* {
            auto* t = graph->add_initializer();
            t->set_name(name);
            t->set_data_type(dtype);
            for (auto d : dims) t->add_dims(d);
            return t;
        };

        // Two FLOAT raw_data initializers, byte-identical, different names.
        const std::string raw_bytes(16, '\x01');
        add_init("w1", onnx::TensorProto_DataType_FLOAT, {4})
            ->set_raw_data(raw_bytes);
        add_init("w2", onnx::TensorProto_DataType_FLOAT, {4})
            ->set_raw_data(raw_bytes);

        // Two INT64 typed-field initializers, value-identical, different names.
        for (const char* name : {"i1", "i2"}) {
            auto* t = add_init(name, onnx::TensorProto_DataType_INT64, {3});
            t->add_int64_data(7);
            t->add_int64_data(8);
            t->add_int64_data(9);
        }

        auto* add1 = graph->add_node();
        add1->set_op_type("Add");
        add1->add_input("w1");
        add1->add_input("w2");
        add1->add_output("wsum");
        auto* add2 = graph->add_node();
        add2->set_op_type("Add");
        add2->add_input("i1");
        add2->add_input("i2");
        add2->add_output("isum");
        graph->add_output()->set_name("wsum");
        graph->add_output()->set_name("isum");

        auto optimized =
            onnx::optimization::Optimize(model, {"eliminate_duplicate_initializer"});
        EXPECT_EQ(optimized.graph().initializer_size(), 2)
            << "trust=" << trust << ": one duplicate removed from each pair";
        for (const auto& node : optimized.graph().node()) {
            EXPECT_EQ(node.input(0), node.input(1))
                << "trust=" << trust << ": " << node.op_type()
                << "'s two inputs were merged to the same initializer";
        }
    }
    onnx::optimization::SetTrustTensorContentHash(true);  // restore the default
}

// HashRawDataBounded (cse_util.h) samples rather than fully hashes raw_data
// past kFullHashLimit, so two distinct large tensors that differ only
// outside the sampled windows can legitimately hash-collide -- this test
// deliberately constructs that worst case and confirms eliminate_duplicate_
// initializer still keeps both (CSETensorCompare's full memcmp is the sole
// source of truth for equality, not the hash), alongside a genuinely
// identical large pair still getting deduped as before.
TEST(OptimizerTest, EliminateDuplicateInitializerLargeRawData) {
    // Larger than kFullHashLimit (4096) and past the point where
    // kMaxSamples (256) windows of kSampleWindow (64) bytes stop covering
    // the whole buffer (n / 256 > 64, i.e. n > 16384), so there are real,
    // unsampled gaps between windows.
    constexpr size_t n = 100000;
    const std::string base(n, 'A');
    std::string differs_in_gap = base;
    // Offset 300 sits in the gap between the first sampled window ([0, 64))
    // and the second (starting at stride = max(64, n/256) = 390), so this
    // byte is never sampled.
    differs_in_gap[300] = 'B';

    ASSERT_EQ(onnx::optimization::HashRawDataBounded(base),
              onnx::optimization::HashRawDataBounded(differs_in_gap))
        << "test assumption: the two buffers hash-collide despite differing";

    onnx::ModelProto model;
    model.set_ir_version(7);
    model.add_opset_import()->set_version(10);
    auto* graph = model.mutable_graph();
    graph->set_name("g");

    auto add_init = [&](const std::string& name, const std::string& raw) {
        auto* t = graph->add_initializer();
        t->set_name(name);
        t->set_data_type(onnx::TensorProto_DataType_UINT8);
        t->add_dims(static_cast<int64_t>(n));
        t->set_raw_data(raw);
    };
    // w1/w2: hash-colliding but byte-distinct -- must NOT be merged.
    add_init("w1", base);
    add_init("w2", differs_in_gap);
    // w3/w4: genuinely identical to each other (but distinct from base) --
    // must still be merged. A different fill byte, not just a different
    // offset, so this pair can't also collide with w1/w2's hash.
    const std::string other(n, 'C');
    add_init("w3", other);
    add_init("w4", other);

    for (const char* name : {"w1", "w2", "w3", "w4"}) {
        auto* n_ = graph->add_node();
        n_->set_op_type("Identity");
        n_->add_input(name);
        n_->add_output(onnx::optimization::Str("out_", name));
        graph->add_output()->set_name(onnx::optimization::Str("out_", name));
    }

    auto optimized =
        onnx::optimization::Optimize(model, {"eliminate_duplicate_initializer"});
    std::unordered_set<std::string> remaining;
    for (const auto& init : optimized.graph().initializer()) {
        remaining.insert(init.name());
    }
    EXPECT_EQ(remaining.count("w1"), 1u) << "hash-colliding but distinct: kept";
    EXPECT_EQ(remaining.count("w2"), 1u) << "hash-colliding but distinct: kept";
    EXPECT_EQ(remaining.count("w3") + remaining.count("w4"), 1u)
        << "genuinely identical: deduped to one";
}

// Regression test for a real model-regression finding: on ~90 real-world
// models, switching CSE's default to trust=true (BLAKE3 digest equality)
// left a handful of transformer models (albert/bert/bart/electra/mvp/
// xcit_*) with MORE remaining nodes than before -- eliminate_common_
// subexpression was merging fewer duplicate Constant nodes than it used to.
// Root cause: for a *typed-field* (non-raw_data) tensor, the old exact
// comparison used std::vector<float>::operator!= (IEEE754 value equality,
// where +0.0f == -0.0f, and std::hash<float> is required to agree), while
// TensorContentDigest hashed the parsed float array's raw bytes verbatim
// (+0.0f and -0.0f have different bit patterns) -- so two Constant nodes
// whose only difference was a signed zero stopped CSE-merging under
// trust=true, and (since CSETensorHash's bucketing always uses the digest,
// regardless of the trust setting) even under trust=false, since they never
// landed in the same hash bucket to begin with. Fixed by canonicalizing
// signed zero before hashing FLOAT/DOUBLE/COMPLEX64/COMPLEX128 typed-field
// values (tensor_content_hash.cc's CanonicalizeZero). This test exercises
// that fix via the real eliminate_common_subexpression pass, under both
// trust settings, since the bug affected both.
TEST(OptimizerTest, SignedZeroConstantCseUnderTrustHash) {
    for (bool trust : {true, false}) {
        onnx::optimization::SetTrustTensorContentHash(trust);

        onnx::ModelProto m;
        m.set_ir_version(7);
        m.add_opset_import()->set_version(13);
        auto* g = m.mutable_graph();
        g->set_name("g");

        auto add_const = [&](const std::string& out_name, float first_elem) {
            auto* node = g->add_node();
            node->set_op_type("Constant");
            node->add_output(out_name);
            auto* attr = node->add_attribute();
            attr->set_name("value");
            attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
            auto* t = attr->mutable_t();
            t->set_data_type(onnx::TensorProto_DataType_FLOAT);
            t->add_dims(2);
            // Typed-field (add_float_data), NOT raw_data -- the case where
            // the old comparison and the digest-based comparison could
            // disagree.
            t->add_float_data(first_elem);
            t->add_float_data(5.0f);
        };
        add_const("c1", 0.0f);
        add_const("c2", -0.0f);
        auto* add = g->add_node();
        add->set_op_type("Add");
        add->add_input("c1");
        add->add_input("c2");
        add->add_output("sum");
        g->add_output()->set_name("sum");

        auto optimized =
            onnx::optimization::Optimize(m, {"eliminate_common_subexpression"});
        ASSERT_EQ(optimized.graph().node_size(), 3)
            << "trust=" << trust
            << ": eliminate_common_subexpression only rewrites uses, "
               "the dangling duplicate Constant node is still present";
        const auto& add_node = optimized.graph().node(2);
        ASSERT_EQ(add_node.op_type(), "Add");
        EXPECT_EQ(add_node.input(0), add_node.input(1))
            << "trust=" << trust
            << ": c1 (+0.0) and c2 (-0.0) should CSE-merge like any other "
               "value-identical Constant pair";
    }
    onnx::optimization::SetTrustTensorContentHash(true);  // restore the default
}
