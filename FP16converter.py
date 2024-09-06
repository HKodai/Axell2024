import onnx
from onnx import helper
from onnx import TensorProto, shape_inference
from onnxconverter_common import float16

model = onnx.load("submit/model.onnx")
model = float16.convert_float_to_float16(model)

# 元の入力ノードを取得
input_tensor = model.graph.input[0]

# FP32に入力タイプを変更
input_tensor.type.tensor_type.elem_type = TensorProto.FLOAT

# FP32からFP16にキャストするノードを作成
cast_node = helper.make_node(
    "Cast", inputs=[input_tensor.name], outputs=["casted_input"], to=TensorProto.FLOAT16
)

# 最初のレイヤーの入力をキャストされたものに変更
for node in model.graph.node:
    for i, input_name in enumerate(node.input):
        if input_name == input_tensor.name:
            node.input[i] = "casted_input"

# キャストノードをモデルの最初に追加
model.graph.node.insert(0, cast_node)

# 型の推論を行い、モデルを保存
model = shape_inference.infer_shapes(model)
onnx.save(model, "submit/model.onnx")
