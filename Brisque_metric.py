from utils import calculate_mscn_coefficients, calculate_pair_product_coefficients
from utils import asymmetric_generalized_gaussian_fit, scale_features
import cv2
from libsvm import svmutil
from itertools import chain
import numpy as np

class Brisque_metric:
    def __init__(self, kernel_size=7, sigma=7/6):
        self.kernel_size = kernel_size
        self.sigma = sigma


    def calculate_brisque_features(self, gray_image):
        def calculate_features(coefficients_name, coefficients, accum=np.array([])):
            alpha, mean, sigma_l, sigma_r = asymmetric_generalized_gaussian_fit(coefficients)

            if coefficients_name == 'mscn':
                var = (sigma_l ** 2 + sigma_r ** 2) / 2
                return [alpha, var]
            
            return [alpha, mean, sigma_l ** 2, sigma_r ** 2]
        
        mscn_coefficients = calculate_mscn_coefficients(gray_image, self.kernel_size, self.sigma)
        coefficients = calculate_pair_product_coefficients(mscn_coefficients)
        
        features = [calculate_features(name, coeff) for name, coeff in coefficients.items()]
        flatten_features = list(chain.from_iterable(features))
        return np.array(flatten_features)

    def calculate_image_quality_score(self, brisque_features):
        model = svmutil.svm_load_model('brisque_svm.txt')
        scaled_brisque_features = scale_features(brisque_features)
        
        x, idx = svmutil.gen_svm_nodearray(
            scaled_brisque_features,
            isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED))
        
        nr_classifier = 1
        prob_estimates = (svmutil.c_double * nr_classifier)()
        
        return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)    

    def get_score(self, gray_image):
        brisque_features = self.calculate_brisque_features(gray_image)    
        downscaled_image = cv2.resize(gray_image, None, fx=1/2, fy=1/2, interpolation = cv2.INTER_CUBIC)
        downscale_brisque_features = self.calculate_brisque_features(downscaled_image)
        brisque_features = np.concatenate((brisque_features, downscale_brisque_features))

        score = self.calculate_image_quality_score(brisque_features)
        return score


if __name__ == '__main__':
    call = Brisque_metric()
    img_fusion = cv2.imread('fusion.png')
    gray_image_fusion = cv2.cvtColor(img_fusion, cv2.COLOR_BGR2GRAY)/255.0
    img_over = cv2.imread('over.jpg')
    gray_image_over = cv2.cvtColor(img_over, cv2.COLOR_BGR2GRAY)/255.0
    img_under = cv2.imread('under.jpg')
    gray_image_under = cv2.cvtColor(img_under, cv2.COLOR_BGR2GRAY)/255.0

    fusion = call.get_score(gray_image_fusion)
    over = call.get_score(gray_image_over)
    under = call.get_score(gray_image_under)

    print('fusion: ', fusion)
    print('over: ', over)
    print('under: ', under)
    