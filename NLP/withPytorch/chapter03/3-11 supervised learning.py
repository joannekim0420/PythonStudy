n_epochs = 10
n_batches = 16
batch_size = 16

for epoch_i in range(n_epochs):
    for batch_i in range(n_batches):
        #step0 prepare data
        x_data, y_data = get_toy_data(batch_size)

        #step1 initialize gradient
        perceptron.zero_grad()

        #step2 forward propagation
        y_pred = perceptron(x_data, apply_sigmoid= True)

        #step3 calculcate loss
        loss = bce_loss(y_pred, y_target)

        #step4 back propagation , compute gradient
        loss.backward()

        #step5 update parameter with optimizer
        optimizer.step()

        
