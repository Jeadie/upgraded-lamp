from enum import Enum
from quora import labelled_data_generator
from embedding import GoogleNegativeNewsEmbeddingWrapper
import tensorflow as tf
import logging

_logger = logging.getLogger()


class ModelMode(Enum): 
    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"

class CNNModel(object): 
    def __init__(self, features, labels, config): 
        """
        Args: 
            features (tf.Tensor) : 
            labels (tf.Tensor)   : 
            config(Dict[str, object]) : Additional configuration to make the model. 

        Returns: 
            Constructor. 

        """
        self.features = features 
        self.labels   = labels
        self.config   = config


    def construct(self): 
        """ Constructs the CNNModel as a untrained computational graph. 

        Returns: 
            (tf.Graph) : 
        """
        
        # Extract model construction parameters or defaults
        filters     = self.config["filters"]       if not None else 32
        kernel      = self.config["kernel_length"] if not None else 50
        activation  = self.config["activation"]    if not None else tf.nn.relu
        pool        = self.config["pool_length"]   if not None else 10
        dense_layer = self.config["dense_layers"]  if not None else [500, 200]

        # Fit three convolutional, maxpooling layers
        conv_1 = tf.layers.conv2d(
                inputs = self.features, 
                filters=filters,
                kernel_size=[kernel, kernel],
                padding="same",
                activation=activation)

        pool_1 = tf.layers.max_pooling2d(
            inputs = conv_1, 
            pool_size = [pool, pool], 
            strides = 2, 

        )

        conv_2 = tf.layers.conv2d(
                inputs = pool_1, 
                filters=filters,
                kernel_size=[kernel, kernel],
                padding="same",
                activation=activation)

        pool_2 = tf.layers.max_pooling2d(
            inputs = conv_2, 
            pool_size = [pool, pool], 
            strides = 2, 
            
        )
    
        conv_3 = tf.layers.conv2d(
                inputs = pool_2, 
                filters=filters,
                kernel_size=[kernel, kernel],
                padding="same",
                activation=activation)

        pool_3 = tf.layers.max_pooling2d(
            inputs = conv_3, 
            pool_size = [pool, pool], 
            strides = 2, 
            
        )

        # Dense layers, followed by a softmax to get probabilities
        prev_layer = [pool_3]
        for size in dense_layer: 
              prev_layer.append(tf.layers.dense(inputs=prev_layer[-1], units=size, activation=activation))

        logits = tf.layers.dense(inputs=prev_layer[-1], units=2)
        return tf.nn.softmax(logits)



class CNNExperiment(object):
    def __init__(self, model, config): 
        """
        Args: 
            config (Dict[str, Object]): User defined configuration to be used in the experiment.

        Returns: 
            Constructor
        """
        self.cnn = model
        self.config = config

        # At "runtime" there will also be references to 
        # self.features
        # self.labels
        # self.mode


    def loss_function(self): 
        """ Defines the loss function to be used in training. 

        Returns: 
            (Func) : loss function to be used in training. 
        """
        return tf.losses.softmax_cross_entropy(
                onehot_labels=tf.one_hot(tf.to_int32(self.labels), 2),
                logits = self.cnn.model
        )

    def evaluation_metrics(self):
        """ Gets the evaluation metrics to be used in training. 

        Returns: 
            (Dict[str, tf.Tensor]) : An evaluation metrics used during training and evaluation
        """
        return {"accuracy" : tf.metrics.accuracy(
                            tf.one_hot(tf.to_int32(self.labels), 2),
                            self.cnn.model
                    )
        }


    def configure_summary(self, loss):
        """
        Set up the experiment summary.

        Args: 
            Loss (tf.Tensor) : the current loss of the experiment 

        Returns: 
            None 
        """
        tf.summary("loss", loss)


    def predictions(self):
        """
        Calculates the predictions from the model in the experiment. 

        Returns: 
            (Dict[str, tf.Tensor]) : 
        """
        return self.cnn.model

    def export_outputs(self):
        """Returns the experiment's export outputs
        
        Returns: 
            Dict[str, tf.Tensor]
        """
        return {}

    def training_optimiser(self, loss):
        """ Configures the training optimiser used in the experiment. 

        Args: 
            loss (Func) : The loss function to optimise. 
            
        Returns: 
            (tf.train.Optimizer) : The configured optimised to use in the experiment. 

        """
        return tf.train.AdamOptimizer(learning_rate = self.config["learning_rate"] if not None else 0.001)

    def model(self, features, labels, mode, params=None, config=None):
        """ Constructs the experiments model for a given experiment mode

        Args: 
            features (tf.Tensor)    : 
            labels (tf.Tensor)      : 
            model  (Type(CNNModel)) :

        Return:
            (tf.EstimatorSpec)
        """
        self.labels   = labels
        self.features = features
        self.mode     = mode
        pu.db
        if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            loss = self.loss_function()
            train_op = self.training_optimiser(loss)

            eval_metrics = self.evaluation_metrics()
            self.configure_summary(loss)
            export_outputs = None

        else:
            loss = None
            train_op = None
            eval_metrics = None
            export_outputs = self.export_outputs()

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metrics,
            predictions=self.predictions(),
            export_outputs=export_outputs,
        )

def run(config): 
    features = tf.FixedLenFeature(shape=[256, 256, 1], dtype=tf.float32)
    labels = tf.FixedLenFeature(shape=[256, 256, 1], dtype=tf.float32)
    config["labels"]   = labels
    config["features"] = features
    tf.logging.set_verbosity(tf.logging.DEBUG)

    model = CNNModel(features, labels, config["model"])
    _logger.info("Model Instantiated.")
    experiment = CNNExperiment(model, config)
    _logger.info("Experiment Instantiated")

    run_config = tf.estimator.RunConfig(
        model_dir=None,
        tf_random_seed=None,
        save_summary_steps=100,
        session_config=None,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=100,
        train_distribute=None,
        device_fn=None,
        protocol=None,
        eval_distribute=None,
        experimental_distribute=None
    )
    estimator = tf.estimator.Estimator(experiment.model, "./", run_config, params=config, warm_start_from=None)
    _logger.info("Esimator Instantiated")

    embedder = GoogleNegativeNewsEmbeddingWrapper("../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin", 
                    word_length = config["input_shape"][1], question_length =config["input_shape"][1])


    train_spec = tf.estimator.TrainSpec(
        input_fn = labelled_data_generator(embedder, config["file_chunk_size"], config["train_file"]), 
        max_steps = config["train_steps"]
    )


    eval_spec = tf.estimator.EvalSpec(
        input_fn = labelled_data_generator(embedder, config["file_chunk_size"], config["test_file"]), 
        steps = config["eval_steps"]
    )
    _logger.info("Training about to commence.")
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__': 
    config = {
        "model" : {
            "filters"       : 32,
            "kernel_length" : 50,
            "activation"    : tf.nn.relu,
            "pool_length"   : 10, 
            "dense_layers"  : [500, 200], 
            }, 
        "train_steps"   : 10, 
        "eval_steps"    : 10,
        "learning_rate" : 0.001, 
        "mode"          : tf.estimator.ModeKeys.TRAIN, 
        "input_shape"   : (50, 300), 
        "file_chunk_size"    : 1000, 
        "train_file"    :  "../input/train.csv", 
        "test_file"     : "../input/test.csv"
    }  
    run(config)