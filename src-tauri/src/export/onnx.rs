use prost::Message;

// ---------------------------------------------------------------------------
// Minimal ONNX schema (field tags match onnx.proto), so no protoc is needed.
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    /// 1 = FLOAT, 2 = INT
    #[prost(int32, tag = "20")]
    pub r#type: i32,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    /// 1 = FLOAT
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TensorTypeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<Dimension>,
}

#[derive(Clone, PartialEq, Message)]
pub struct Dimension {
    #[prost(int64, optional, tag = "1")]
    pub dim_value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    pub dim_param: Option<String>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    None,
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
}

/// One dense layer.
pub struct DenseLayerSpec<'a> {
    pub in_features: usize,
    pub out_features: usize,
    /// Flat weights: row-major `[in, out]`, or `[out, in]` when `transposed`.
    pub weights: &'a [f32],
    pub bias: &'a [f32],
    pub activation: Activation,
    pub transposed: bool,
}

/// Anything that can describe itself as a stack of dense layers.
pub trait OnnxExportable {
    fn onnx_layers(&self) -> Result<Vec<DenseLayerSpec<'_>>, String>;
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn tensor_info(name: &str, features: usize) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1, // FLOAT
                shape: Some(TensorShapeProto {
                    dim: vec![
                        Dimension {
                            dim_value: None,
                            dim_param: Some("batch".into()),
                        },
                        Dimension {
                            dim_value: Some(features as i64),
                            dim_param: None,
                        },
                    ],
                }),
            }),
        }),
    }
}

/// Builds the bytes of an `.onnx` file: `input -> (Gemm -> activation)* -> output`.
pub fn build_onnx<T: OnnxExportable>(model: &T) -> Result<Vec<u8>, String> {
    let layers = model.onnx_layers()?;
    if layers.is_empty() {
        return Err("Cannot export an empty model to ONNX.".into());
    }

    // Layers must chain: each layer's input size equals the previous output size.
    for i in 1..layers.len() {
        if layers[i].in_features != layers[i - 1].out_features {
            return Err(format!(
                "Layer {i} expects {} inputs but layer {} produces {} outputs.",
                layers[i].in_features,
                i - 1,
                layers[i - 1].out_features
            ));
        }
    }

    let mut nodes = Vec::new();
    let mut inits = Vec::new();
    let mut prev = "input".to_string();
    let last = layers.len() - 1;

    for (i, l) in layers.iter().enumerate() {
        if l.weights.len() != l.in_features * l.out_features {
            return Err(format!(
                "Layer {i}: weight length {} does not match {}x{}",
                l.weights.len(),
                l.in_features,
                l.out_features
            ));
        }
        if l.bias.len() != l.out_features {
            return Err(format!(
                "Layer {i}: bias length does not match out_features"
            ));
        }

        let w_name = format!("W{i}");
        let b_name = format!("B{i}");

        let w_dims = if l.transposed {
            vec![l.out_features as i64, l.in_features as i64]
        } else {
            vec![l.in_features as i64, l.out_features as i64]
        };

        inits.push(TensorProto {
            dims: w_dims,
            data_type: 1,
            name: w_name.clone(),
            raw_data: f32_bytes(l.weights),
        });
        inits.push(TensorProto {
            dims: vec![l.out_features as i64],
            data_type: 1,
            name: b_name.clone(),
            raw_data: f32_bytes(l.bias),
        });

        let has_act = !matches!(l.activation, Activation::None);
        let gemm_out = if i == last && !has_act {
            "output".to_string()
        } else {
            format!("fc{i}")
        };

        // Y = X * W (+ B). transB = 1 when W is stored as [out, in].
        nodes.push(NodeProto {
            input: vec![prev.clone(), w_name, b_name],
            output: vec![gemm_out.clone()],
            name: format!("gemm{i}"),
            op_type: "Gemm".into(),
            attribute: if l.transposed {
                vec![AttributeProto {
                    name: "transB".into(),
                    f: 0.0,
                    i: 1,
                    r#type: 2,
                }]
            } else {
                vec![]
            },
        });

        let mut out_name = gemm_out.clone();
        if has_act {
            let (op, attrs) = match l.activation {
                Activation::Relu => ("Relu", vec![]),
                Activation::Sigmoid => ("Sigmoid", vec![]),
                Activation::Tanh => ("Tanh", vec![]),
                Activation::Softmax => (
                    "Softmax",
                    vec![AttributeProto {
                        name: "axis".into(),
                        f: 0.0,
                        i: -1,
                        r#type: 2,
                    }],
                ),
                Activation::None => unreachable!(),
            };
            out_name = if i == last {
                "output".to_string()
            } else {
                format!("act{i}")
            };
            nodes.push(NodeProto {
                input: vec![gemm_out],
                output: vec![out_name.clone()],
                name: format!("act{i}"),
                op_type: op.into(),
                attribute: attrs,
            });
        }
        prev = out_name;
    }

    let model_proto = ModelProto {
        ir_version: 8,
        producer_name: "tensorloom".into(),
        producer_version: env!("CARGO_PKG_VERSION").into(),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            name: "tensorloom_mlp".into(),
            node: nodes,
            initializer: inits,
            input: vec![tensor_info("input", layers[0].in_features)],
            output: vec![tensor_info("output", layers[last].out_features)],
        }),
    };

    Ok(model_proto.encode_to_vec())
}
