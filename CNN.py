from enum import Enum

class ModelMode(Enum): 
    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"

class CNNModel(object): 
    def __init__(self, features, labels, mode): 
        """
        Args: 
            features (tf.Tensor) : 
            labels (tf.Tensor)   : 
            mode (ModelMode)     : 

        Returns: 
            Constructor. 

        """
        pass

    def construct(self): 
        """ Constructs the CNNModel as a untrained computational graph. 

        Returns: 
            (tf.Graph) : 
        """

    



class CNNExperiment(object):
    def __init__(self, config): 
        """
        Args: 
            config (Dict[str, Object]): User defined configuration to be used in the experiment.

        Returns: 
            Constructor
        """
        pass

    def model(self, features, labels, model=CNNModel): 
        """ Constructs the overall spec of the model, for the given model and mode. 

        Args: 
            features (tf.Tensor)    : 
            labels (tf.Tensor)      : 
            model  (Type(CNNModel)) : 

        Returns: 
            tf.estimator.EstimatorSpec
        """

    def loss_function(self): 
        """ Defines the loss function to be used in training. 

        Returns: 
            (Func) : loss function to be used in training. 
        """

    def evaluation_metrics(self):
        """ Gets the evaluation metrics to be used in training. 

        Returns: 
            (Dict[str, tf.Tensor]) : An evaluation metrics used during training and evaluation
        """
        return {}

    def configure_summary(self):
        """
        Set up the experiment summary.

        Returns: 
            None 
        """
        pass

    def predictions(self):
        """
        Calculates the predictions from the model in the experiment. 

        Returns: 
            (Dict[str, tf.Tensor]) : 
        """
        return {}

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
        return None

    def model(self, features, labels, mode):
        """ Constructs the experiments model for a given experiment mode

        Args: 
            features (tf.Tensor)    : 
            labels (tf.Tensor)      : 
            model  (Type(CNNModel)) :

        Return:
            (tf.EstimatorSpec)
        """
    