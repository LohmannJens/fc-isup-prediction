import tensorflow as tf


def isup_cce_loss_wrap():
    """
    Wrapper function to calculate the categorical accuracy as needed for eCaReNet
    
    :return: calculated loss
    """
    def isup_cce_loss(y_true, y_pred):
        # decide which probability to use based on y_true
        result = tf.where(y_true == 1.0, y_pred, 1.0 - y_pred)
        # clip values to avoid 0.0
        result_clipped = tf.clip_by_value(result, 1e-7, 1 - 1e-3)
        # calculate log
        log_result = tf.math.log(result_clipped)
        # calculate sum and change to positive value
        sum_log_result = tf.reduce_sum(log_result, axis=1) * -1
        return sum_log_result
    return isup_cce_loss