import torch
from torchvision import models

class Img2Vec():

    def __init__(self, model_path='./fine_tuning_dict.pt'):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            self.model = torch.load(model_path) # because the model was trained on a cuda machine
        else:
            self.model = torch.load(model_path, map_location='cpu')

        self.extraction_layer = self.model._modules.get('avgpool')
        self.layer_output_size = 2048

        self.model = self.model.to(self.device)
        self.model.eval()


    def get_vec(self, image):

        image = image.to(self.device)

        num_imgs = image.size(0)

        my_embedding = torch.zeros(num_imgs, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        return my_embedding.view(num_imgs, -1)

def extract_feature():
    # mymodel = VGG19_features(trained_model)
    # loader = transforms.Compose([transforms.ToTensor()])
    numbers = re.compile(r'(\d+)')

    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    vdo = '1'
    capt = sorted(paths.list_images('/home/morteza/Videos/frames_test/'+vdo+'/4ps'), key = numericalSort)
    segm = 0
    extracted_features = list()
    while segm<len(capt):
        t1 = time.time()
        img = Image.open(capt[segm])#cv2.imread('/home/morteza/Videos/traffic camera/keyframes/1/seg0.jpg')            
        img = img.resize((960,960))
        img = loader(img).float()
        img = Variable(img)
        img = img.unsqueeze(0)            
        # pred = trained_model(img)
        # print(pred)        
        ### Loading the model and feeding the image to get features in the layer whose output is our interest        
        output = get_vec(img)
        features = (output.data).cpu().numpy() #converting the output of the layer to the numpy array
        # features = features.flatten()        
        extracted_features.append(features.flatten())
        print("proc time: ", time.time()-t1)
        segm += 1
    pickle.dump( extracted_features, open( '/home/morteza/Videos/frames_test/'+vdo+'.p', "wb" ) )