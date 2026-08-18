// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <onnxoptimizer/optimize.h>
#include <onnx/defs/parser.h>

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

    onnx::ModelProto optimized_model = onnx::optimization::PrepareOutput(model);
    onnx::ExportModelProto(&optimized_model, graph);

    ASSERT_EQ(optimized_model.graph().node().size(), 1);
    ASSERT_EQ(optimized_model.graph().node()[0].op_type(), "Identity");
}
