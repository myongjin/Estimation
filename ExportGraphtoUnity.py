from tensorflow.python.tools import freeze_graph

model_path = './checkpoints'
last_checkpoint= 'my_checkpoint.data-00000-of-00001'
freeze_graph.freeze_graph(input_graph = model_path +'/checkpoint.ckpt',
              input_binary = True,
              input_checkpoint = last_checkpoint,
              output_node_names = "action",
              output_graph = model_path +'/your_name_graph.bytes' ,
              clear_devices = True, initializer_nodes = "",input_saver = "",
              restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0")

