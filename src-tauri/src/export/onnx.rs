use prost::Message;
use onnx_protobuf::{
    graph_proto::{GraphProto, ValueInfoProto},
    model_proto::ModelProto,
    opset_import::OpsetImport,
    tensor_proto::{TensorProto, DataProto},
    node_proto::NodeProto,
};
use tensorloom_core::model::PortableModel;

pub fn export_to_onnx(model: &PortableModel) -> Result<Vec<u8>, String> {
    let mut graph = GraphProto::default();
    let mut initializers = Vec::new();

    // 1. Define Input
    let n_in = model.input_size().ok_or("Model has no layers")?;
    let input_name = "input".to_string();
    graph.input.push(ValueInfoProto {
        name: Some(input_name.clone()),
        ..Default::default()
    });

    let mut current_input = input_name;

    // 2. Build Layer Chain
    for (i, layer) in model.layers.iter().enumerate() {
        let (n_in_layer, n_out_layer) = layer.shape;
        let weight_name = format!("weight_{}", i);
        let bias_name = format!("bias_{}", i);
        let gemm_out = format!("gemm_out_{}", i);
        let act_out = format!("act_out_{}", i);

        // Weights Initializer [n_in, n_out]
        initializers.push(TensorProto {
            name: Some(weight_name.clone()),
            data: Some(DataProto {
                float_data: layer.weights.clone(),
                ..Default::default()
            }),
            ..Default::default()
        });

        // Bias Initializer [n_out]
        initializers.push(TensorProto {
            name: Some(bias_name.clone()),
            data: Some(DataProto {
                float_data: layer.bias.clone(),
                ..Default::default()
            }),
            ..Default::default()
        });

        // Gemm Node: Y = XW + B
        graph.node.push(NodeProto {
            input: vec![current_input.clone(), weight_name, bias_name],
            output: vec![gemm_out.clone()],
            op_type: "Gemm".to_string(),
            ..Default::default()
        });

        // Activation Node
        match layer.activation.as_str() {
            "relu" => {
                graph.node.push(NodeProto {
                    input: vec![gemm_out],
                    output: vec![act_out.clone()],
                    op_type: "Relu".to_string(),
                    ..Default::default()
                });
                current_input = act_out;
            }
            "sigmoid" => {
                graph.node.push(NodeProto {
                    input: vec![gemm_out],
                    output: vec![act_out.clone()],
                    op_type: "Sigmoid".to_string(),
                    ..Default::default()
                });
                current_input = act_out;
            }
            "tanh" => {
                graph.node.push(NodeProto {
                    input: vec![gemm_out],
                    output: vec![act_out.clone()],
                    op_type: "Tanh".to_string(),
                    ..Default::default()
                });
                current_input = act_out;
            }
            _ => {
                // Identity / Linear
                current_input = gemm_out;
            }
        }
    }

    // 3. Define Output
    graph.output.push(ValueInfoProto {
        name: Some(current_input),
        ..Default::default()
    });

    graph.initializer = initializers;

    // 4. Final Model
    let model_proto = ModelProto {
        ir_version: 7,
        opset_import: vec![OpsetImport {
            domain: "".to_string(),
            version: 13,
        }],
        graph: Some(graph),
        ..Default::default()
    };

    let mut buf = Vec::new();
    model_proto.encode(&mut buf).map_err(|e| format!("Failed to encode ONNX model: {e}"))?;
    Ok(buf)
}
