from tensorflow.keras.layers import Conv1D as _Conv1D
from tensorflow.keras.layers import Conv2D as _Conv2D
from tensorflow.keras.layers import MaxPooling2D as _MaxPooling2D
from tensorflow.keras.layers import Dense as _Dense
from tensorflow.keras.layers import Embedding as _Embedding
from tensorflow.keras.layers import LSTM as _LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout as _Dropout
from tensorflow.keras.layers import BatchNormalization as _BatchNormalization
from tensorflow.keras.layers import TimeDistributed as _TimeDistributed
from tensorflow.keras.layers import Activation as _Activation
from tensorflow.keras.layers import Flatten as _Flatten
from tensorflow.keras.layers import Reshape as _Reshape
from tensorflow.keras import regularizers

from numpy import log2

from autogoal.grammar import (
    BooleanValue,
    CategoricalValue,
    DiscreteValue,
    ContinuousValue,
)
from autogoal.utils import nice_repr


@nice_repr
class Seq2SeqLSTM(_LSTM):
    def __init__(
        self,
        units: DiscreteValue(32, 1024),
        activation_fn: CategoricalValue("tanh", "sigmoid", "relu", "linear"),
        recurrent_activation_fn: CategoricalValue("tanh", "sigmoid", "relu", "linear"),
        dropout: ContinuousValue(0, 0.5),
        recurrent_dropout: ContinuousValue(0, 0.5),
        **kwargs
    ):
        super().__init__(
            units=units,
            activation=activation_fn,
            recurrent_activation=recurrent_activation_fn,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            **kwargs
        )

        self.activation_fn = activation_fn
        self.recurrent_activation_fn = recurrent_activation_fn

    def get_config(self):
        config = super().get_config()
        config["activation_fn"] = self.activation_fn
        config["recurrent_activation_fn"] = self.recurrent_activation_fn
        config.pop("return_sequences", None)
        config.pop("activation", None)
        config.pop("recurrent_activation", None)
        return config


@nice_repr
class Seq2VecLSTM(_LSTM):
    def __init__(
        self,
        units: DiscreteValue(32, 1024),
        activation_fn: CategoricalValue("tanh", "sigmoid", "relu", "linear"),
        recurrent_activation_fn: CategoricalValue("tanh", "sigmoid", "relu", "linear"),
        dropout: ContinuousValue(0, 0.5),
        recurrent_dropout: ContinuousValue(0, 0.5),
        **kwargs
    ):
        super().__init__(
            units=units,
            activation=activation_fn,
            recurrent_activation=recurrent_activation_fn,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=False,
            **kwargs
        )

        self.activation_fn = activation_fn
        self.recurrent_activation_fn = recurrent_activation_fn

    def get_config(self):
        config = super().get_config()
        config["activation_fn"] = self.activation_fn
        config["recurrent_activation_fn"] = self.recurrent_activation_fn
        config.pop("return_sequences", None)
        config.pop("activation", None)
        config.pop("recurrent_activation", None)
        return config


@nice_repr
class Seq2SeqBiLSTM(Bidirectional):
    def __init__(
        self,
        merge_mode: CategoricalValue("sum", "mul", "concat", "ave"),
        units: DiscreteValue(32, 1024),
        activation_fn: CategoricalValue("tanh", "sigmoid", "relu", "linear"),
        recurrent_activation_fn: CategoricalValue("tanh", "sigmoid", "relu", "linear"),
        dropout: ContinuousValue(0, 0.5),
        recurrent_dropout: ContinuousValue(0, 0.5),
        **kwargs
    ):
        super().__init__(
            layer=_LSTM(
                units=units,
                activation=activation_fn,
                recurrent_activation=recurrent_activation_fn,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,
            ),
            merge_mode=merge_mode,
            **kwargs
        )

        self.activation_fn = activation_fn
        self.recurrent_activation_fn = recurrent_activation_fn

    def get_config(self):
        config = super().get_config()
        config["units"] = self.layer.units
        config["activation_fn"] = self.activation_fn
        config["recurrent_activation_fn"] = self.recurrent_activation_fn
        config["dropout"] = self.layer.dropout
        config["recurrent_dropout"] = self.layer.recurrent_dropout
        config.pop("layer", None)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@nice_repr
class Seq2VecBiLSTM(Bidirectional):
    def __init__(
        self,
        merge_mode: CategoricalValue("sum", "mul", "concat", "ave"),
        units: DiscreteValue(32, 1024),
        activation_fn: CategoricalValue("tanh", "sigmoid", "relu", "linear"),
        recurrent_activation_fn: CategoricalValue("tanh", "sigmoid", "relu", "linear"),
        dropout: ContinuousValue(0, 0.5),
        recurrent_dropout: ContinuousValue(0, 0.5),
        **kwargs
    ):
        super().__init__(
            layer=_LSTM(
                units=units,
                activation=activation_fn,
                recurrent_activation=recurrent_activation_fn,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=False,
            ),
            merge_mode=merge_mode,
            **kwargs
        )

        self.activation_fn = activation_fn
        self.recurrent_activation_fn = recurrent_activation_fn

    def get_config(self):
        config = super().get_config()
        config["units"] = self.layer.units
        config["activation_fn"] = self.activation_fn
        config["recurrent_activation_fn"] = self.recurrent_activation_fn
        config["dropout"] = self.layer.dropout
        config["recurrent_dropout"] = self.layer.recurrent_dropout
        config.pop("layer", None)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@nice_repr
class Reshape2D(_Reshape):
    def __init__(self, **kwargs):
        super().__init__(target_shape=(-1, 1), **kwargs)

    def get_config(self):
        config = super().get_config()
        config.pop("target_shape", None)
        return config


@nice_repr
class Embedding(_Embedding):
    def __init__(self, output_dim: DiscreteValue(32, 128), **kwargs):
        super().__init__(input_dim=1000, output_dim=output_dim, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.pop("input_dim", None)
        return config


@nice_repr
class Dense(_Dense):
    def __init__(self, units: DiscreteValue(128, 1024), **kwargs):
        super().__init__(units=units, **kwargs)


@nice_repr
class Conv1D(_Conv1D):
    def __init__(
        self,
        filters: DiscreteValue(2, 8),
        kernel_size: CategoricalValue(3, 5, 7),
        **kwargs
    ):
        super().__init__(
            filters=2 ** filters, kernel_size=kernel_size, padding="causal", **kwargs
        )

    def get_config(self):
        config = super().get_config()
        config["filters"] = log2(config["filters"])
        config.pop("padding", None)
        return config


@nice_repr
class Conv2D(_Conv2D):
    def __init__(
        self,
        filters: DiscreteValue(2, 8),
        kernel_size: CategoricalValue(3, 5, 7),
        l1: ContinuousValue(0, 1e-3),
        l2: ContinuousValue(0, 1e-3),
        **kwargs
    ):
        self.l1 = l1
        self.l2 = l2
        super().__init__(
            filters=2 ** filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            padding="same",
            data_format="channels_last",
            **kwargs
        )

    def get_config(self):
        config = super().get_config()
        config["l1"] = self.l1
        config["l2"] = self.l2
        config["kernel_size"] = self.kernel_size[0]
        config["filters"] = log2(config["filters"])
        config.pop("kernel_regularizer", None)
        config.pop("padding", None)
        config.pop("data_format", None)
        return config


@nice_repr
class MaxPooling2D(_MaxPooling2D):
    def __init__(self, **kwargs):
        super().__init__(data_format="channels_last", padding="same", **kwargs)

    def get_config(self):
        config = super().get_config()
        config.pop("data_format", None)
        config.pop("padding", None)
        return config


@nice_repr
class TimeDistributed(_TimeDistributed):
    def __init__(self, layer: Dense, **kwargs):
        super().__init__(layer, **kwargs)


@nice_repr
class Dropout(_Dropout):
    def __init__(self, rate: ContinuousValue(0, 0.5), **kwargs):
        super().__init__(rate=rate, **kwargs)


@nice_repr
class BatchNormalization(_BatchNormalization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@nice_repr
class Activation(_Activation):
    def __init__(
        self,
        function: CategoricalValue(
            "elu",
            "selu",
            "relu",
            "tanh",
            "sigmoid",
            "hard_sigmoid",
            "exponential",
            "linear",
        ),
        **kwargs
    ):
        self.function = function
        super().__init__(activation=function, **kwargs)

    def get_config(self):
        config = super().get_config()
        config["function"] = config["activation"]
        config.pop("activation", None)
        return config


@nice_repr
class Flatten(_Flatten):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
