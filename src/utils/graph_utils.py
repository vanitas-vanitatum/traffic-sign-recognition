import tensorflow as tf


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
    return graph


def load_model_from_estimator(model_path: str, session: tf.Session):
    tf.saved_model.loader.load(
        session,
        ["serve"],
        model_path,
        import_scope=None,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model_file")
    args = parser.parse_args()
    #
    # load_graph(args.model_file)
    with tf.Session() as sess:
        load_model_from_estimator(
            args.model_file, sess)
        print(sess.graph_def)
