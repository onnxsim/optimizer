// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <onnxoptimizer/optimize.h>
#include <onnx/defs/parser.h>

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
