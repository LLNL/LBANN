import lbann
import cosmoflow_network_architectures
import math

def construct_cosmoflow_model(parallel_strategy,
                              local_batchnorm,
                              input_width,
                              num_secrets,
                              use_batchnorm,
                              num_epochs,
                              learning_rate,
                              min_distconv_width):

    # Construct layer graph
    universes = lbann.Input(data_field='samples')
    secrets = lbann.Input(data_field='responses')
    statistics_group_size = 1 if local_batchnorm else -1
    preds = cosmoflow_network_architectures.CosmoFlow(
        input_width=input_width,
        output_size=num_secrets,
        use_bn=use_batchnorm,
        bn_statistics_group_size=statistics_group_size)(universes)
    mse = lbann.MeanSquaredError([preds, secrets])
    obj = lbann.ObjectiveFunction([mse])
    layers = list(lbann.traverse_layer_graph([universes, secrets]))

    # Set parallel_strategy
    if parallel_strategy is not None:
        depth_groups = int(parallel_strategy['depth_groups'])
        min_distconv_width = max(depth_groups, min_distconv_width)
        last_distconv_layer = int(math.log2(input_width)
                                  - math.log2(min_distconv_width) + 1)
        for layer in layers:
            if layer == secrets:
                continue

            if f'pool{last_distconv_layer}' in layer.name or 'fc' in layer.name:
                break

            layer.parallel_strategy = parallel_strategy

    # Set up model
    metrics = [lbann.Metric(mse, name='MSE', unit='')]
    callbacks = [
        lbann.CallbackPrint(),
        lbann.CallbackTimer(),
        lbann.CallbackGPUMemoryUsage(),
        lbann.CallbackPrintModelDescription(),
        lbann.CallbackDumpOutputs(
            directory='dump_acts/',
            layers=' '.join([preds.name, secrets.name]),
            execution_modes='test'
        ),
        lbann.CallbackProfiler(skip_init=True),
        lbann.CallbackLinearGrowthLearningRate(target=learning_rate, num_epochs=5),
        lbann.CallbackSetLearningRate(step=32, val=0.25 * learning_rate),
        lbann.CallbackSetLearningRate(step=64, val=0.125 * learning_rate),
    ]
    # # TODO: Use polynomial learning rate decay (https://github.com/LLNL/lbann/issues/1581)
    # callbacks.append(lbann.CallbackPolyLearningRate(
    #     power=1.0,
    #     num_epochs=100,
    #     end_lr=1e-7))
    return lbann.Model(
        epochs=num_epochs,
        layers=layers,
        objective_function=obj,
        metrics=metrics,
        callbacks=callbacks
    )
