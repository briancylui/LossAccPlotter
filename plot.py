from laplotter import LossAccPlotter

num_classes  = 40

log_suffixes = ['_150_150_15', '_150_300_30', '_200_400_5', '_300_60_40.txt', '_300_60_45', '_60_300_30', '_60_300_35', '_60_300_40', '_60_350_35']

for i in range(len(log_suffixes)):
    suffix = log_suffixes[i]
    num_classes, num_train_per_class, batch_size = None, None, None
    start1 = suffix.find('_') + 1
    end1 = suffix.find('0_') + 1
    start2 = end1 + 1
    end2 = suffix.rfind('0_') + 1
    start3 = end2 + 1
    end3 = suffix.rfind('.')
    
    num_classes = suffix[start1:end1]
    num_train_per_class = suffix[start2:end2]
    batch_size = suffix[start3:] if end3 == -1 else suffix[start3:end3]

    with open("/home/jaebumlee94/deepclip/bumbumnet/log" + suffix) as text:
        loss = []
        Epoch = []
        train_acc = []
        val_acc_top1 = []
        val_acc_top5 = []
        count_loss = 0
        count_acc = 0
        for line in text:
            if line.startswith("Epoch"):
                loss.append(float(line.rstrip('\n').split(": ")[-1]))
                count_loss += 1
            elif line.startswith("Current "):
                loss.append(float(line[line.find(".")-1:-1].rstrip()))
                count_loss += 1
            elif line.startswith("Train"):
                train_acc.append(float(line.rstrip('\n').split("(")[-1][:-1]))
                count_acc += 1
            elif line.startswith("Val"):
                val_acc_top1.append(float(line.rstrip('\n').replace('; ', '( ').split("(")[1][:-1]))
                val_acc_top5.append(float(line.rstrip('\n').split("(")[-1][:-1]))
                
            while count_loss>count_acc:
                train_acc.append(None)
                val_acc_top1.append(None)
                val_acc_top5.append(None)
                count_acc += 1
    print("Start plotting")
    # print(len(loss))
    # print(train_acc)
    # print(val_acc_top1)
    # print(val_acc_top5)

    plot_title = 'Loss and Accuracy Plotted Using ' + num_classes + ' Keywords (' + num_train_per_class + ' training videos each) and minibatch size of ' + batch_size, save_to_filepath=num_classes + '_' + num_train_per_class + '_' + batch_size + '.png'
    
    plotter = LossAccPlotter(title=plot_title, save_to_filepath=num_classes + '_' + num_train_per_class + '_' + batch_size + '.png', show_regressions=False, show_averages=False, show_loss_plot=True, show_acc_plot=True, show_plot_window=True, x_label="Epoch")
            
    for epoch in range(len(loss)):
        plotter.add_values(epoch, loss_train = loss[epoch], acc_train = train_acc[epoch], acc_val = val_acc_top1[epoch])
        
    plotter.block()
            
    print("Finished plotting")
