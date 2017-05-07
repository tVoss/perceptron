import pickle
import random

# Global parameters
epochs = 2
weights = [[0 for _ in range(784)] for _ in range(10)]
biases = [0 for _ in range(10)]

def main():
    print 'Loading data...'
    # Load pickled data for reading in binary mode
    training_data = pickle.load(open('train.p', 'rb'))
    testing_data = pickle.load(open('test.p', 'rb'))

    print 'Training for', epochs, 'epochs...'

    for e in range(epochs):

        # Shuffle the examples and set alpha for this epoch
        random.shuffle(training_data)
        alpha = float(epochs - e) / float(epochs)

        # Train on the first 672 items, test on the remaining
        train(training_data[:4000], alpha)
        percent = test(training_data[4000:])
        print 'Epoch', e + 1, 'was', percent, '% correct'

    print 'Testing...'
    # Used trained perceptron on testing data
    percent = test(testing_data)
    print percent, '% correct'

def train(data, alpha):
    for i in range(len(data)):
        # Go through each image and the number it's supposed to be
        image, num = data[i]
        for w in xrange(10):

            # For each number perceptron, test the result
            y = dot(weights[w], image) + biases[w]
            correct = (y > 0) == (num == w)

            if not correct:
                # When wrong, update the weights
                weights[w] = sub(weights[w], image, alpha) if y > 0 else add(weights[w], image, alpha)
                biases[w] = biases[w] - alpha if y > 0 else biases[w] + alpha

def test(data):
    correct = 0
    for i in range(len(data)):
        image, num = data[i]

        # Test all perceptrons at once
        output = [dot(weights[w], image) for w in range(10)]
        # Take one with highest confidence
        guess = output.index(max(output))
        if guess == num:
            correct += 1

    # Return percent correct
    return float(correct) / float(len(data)) * 100

def add(w, x, a = 1):
    # Helper for adding arrays with weight
    return [w[i] + x[i] * a for i in range(len(x))]

def sub(w, x, a = 1):
    # Helper for subtracting arrays with weight
    return [w[i] - x[i] * a for i in range(len(x))]

def dot(w, x):
    # Helper for getting the dot product of two arrays
    sum = 0
    for i in xrange(len(x)):
        sum += w[i] * x[i]
    return sum

if __name__ == '__main__':
    main()
