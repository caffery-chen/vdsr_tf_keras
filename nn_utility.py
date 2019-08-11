import os
import tensorflow as tf
import json

def train_model(model, training_data, optimizer):
    # GPU resource configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Callbacks
    checkpoint_path = "./training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.TensorBoard(log_dir='./log'),
        # Create checkpoint callback
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=5)
    ]

    # session to config the resource and do weights initialization
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())

    # save model and checkpoints
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    json_string = model.to_json()
    with open('./training/personal.json', 'w') as json_file:
        json.dump(json_string, json_file)
    model.save_weights(checkpoint_path.format(epoch=0))

    # train
    model.fit(training_data['train_data'], training_data['train_label'], epochs=100, batch_size=128,
              callbacks=callbacks,validation_data=(training_data['val_data'], training_data['val_label']))

    # evaluate
    print('\n# Evaluate on test data')
    results = model.evaluate(training_data['test_data'], training_data['test_label'], batch_size=128)
    print('test loss, test acc:', results)