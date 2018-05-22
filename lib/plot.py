import matplotlib.pyplot as plt

# plots the training loss as a graph
#
# @input history
# The history object as returned by Keras' model.fit() method
# @input show (True)
# Boolean if the plot showed be displayed or not
# @input save_file (None)
# If specified the plot will be saved as a png file with that name
#
# @return void
#
def plot_training_loss(history, show=True, save_file=None):
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # Show the plot
    if(show):
        plt.show()
    # Save the plot as a file
    if(save_file):
        plt.savefig(save_file, format="png")
    # clear the plot
    plt.clf()


# plots the training accuracy as a graph
#
# @input history
# The history object as returned by Keras' model.fit() method
# @input show (True)
# Boolean if the plot showed be displayed or not
# @input save_file (None)
# If specified the plot will be saved as a png file with that name
#
# @return void
#
def plot_training_accuracy(history, show=True, save_file=None):
    plt.plot(history.history['acc'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    # Show the plot
    if(show):
        plt.show()
    # Save the plot as a file
    if(save_file):
        plt.savefig(save_file, format="png")
    # clear the plot
    plt.clf()
