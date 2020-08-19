from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

class PredictDigit:

    def __init__(self):
        self.model = load_model('final_model.h5')
        self.prev_pred = None

    def load_image(self, filename):
        img = load_img(filename, grayscale=True, target_size=(28, 28))
        img = img_to_array(img)
        print(img.shape)
        # reshape into a single sample with 1 channel
        img = img.reshape(1, 28, 28, 1)
        # prepare pixel data
        img = img.astype('float32')
        img = img / 255.0
        return img

    def prep_img_arr(self, arr):
        img = arr.reshape(1, 28, 28, 1)
        # prepare pixel data
        img = img.astype('float32')
        img = img / 255.0
        return img


    def run_preds(self, arr=None):
        # load the image
        # img = self.load_image('sample_test_image.png')

        # Load the array
        img = self.prep_img_arr(arr)

        # predict the class
        digit = self.model.predict_classes(img)[0]
        if self.prev_pred is None or self.prev_pred != digit:
            print(digit)
            self.prev_pred = digit

if __name__ == '__main__':
    obj = PredictDigit()
    obj.run_preds()