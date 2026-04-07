import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='SoilClassML')
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    The modulating factor (1 - p_t)^gamma down-weights easy examples
    so the optimiser focuses on hard / misclassified samples.

    Args:
        gamma (float): Focusing parameter. gamma=0 reduces to cross-entropy.
        alpha (float): Overall loss scale factor.
    """
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        loss = -self.alpha * focal_weight * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config
