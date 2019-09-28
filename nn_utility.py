import os
import tensorflow as tf
import json
from customize_callbacks.ConstellationCallback import ConstellationCallbacks

def train_model(training_data, test_data, val_data, model, optimizer, epoch, batch_size, log_dir):
    # GPU resource configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Callbacks
    checkpoint_path = "%s/cp-{epoch:04d}.ckpt" % log_dir
    checkpoint_dir = os.path.dirname(checkpoint_path)

    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        #tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True,write_grads=True, write_images = False),
        # Create checkpoint callback
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=50),
        ConstellationCallbacks(logdir = log_dir, period = 5, val_data = val_data['val_data'][25:100,:,:,:])
    ]

    # session to config the resource and do weights initialization
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())

    # save model and checkpoints
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    json_string = model.to_json()
    with open('%s/model.json' % log_dir, 'w') as json_file:
        json.dump(json_string, json_file)
    model.save_weights(checkpoint_path.format(epoch=0))

    # train
    model.fit(training_data['train_data'], training_data['train_label'], epochs=epoch, batch_size=batch_size, shuffle = True,
              callbacks=callbacks,validation_data=(val_data['val_data'], val_data['val_label']))

    # evaluate
    print('\n# Evaluate on test data')
    results = model.evaluate(test_data['test_data'], test_data['test_label'], batch_size=batch_size)
    print('test loss, test acc:', results)
