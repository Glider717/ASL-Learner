<div align="center">
	<h1>ASL Learner</h1>
</div>

<br/>
<div align="center" style="display:grid; 
            justify-content: center;
            padding:15px 0 30px 0">
    <div>
        <img style="margin-right: 35px; width: 300px; height: 250px" src = "https://3.files.edl.io/5047/21/09/09/154508-86885ac7-894b-431b-8f3a-da44043deb9e.jpg" />
	    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </div>
</div>
<br/>


## Description
ASL Learner is a machine learning-based website. We have created this application for people who are interested in learning American Sign language. Through this app, a user can learn how to spell each letter in American Sign language. This app is very useful for people with speaking disabilities and hearing disabilities. This website will teach sign language to people with hearing disability, which will allow them to communication with others. This app will also help people who want to pursue any career to help the hearing-disabled person. Our website is very user-friendly and eye-catching. It is very easy to interact with. We have pages on the website but in the beginning, a user will be greeted with the home page. Then, they can move to the next page where they will find many words on a different levels. The user can choose any word from those levels and it will take them to the page. On that page, the user will try to mimic the sign language of each letter of that chosen word. When the user is done, the website will bring them back to the word choice page. There is also a sign in feature which saves a user’s progress which they can access through the info page.

## Design and Experiments 

### Software Design

### ML Model Design
The ML model used for gesture classification is a modified ResNet152. The output layer of the model was modified to produce 26 classes which correspond to 26 letters in the alphabet. The model was trained for one epoch with ResNet152 layers frozen and then for an additional epoch with all layers unfrozen. 

#### Hyperparameters
```python
epochs = 1
max_lr = 1e-4
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
```

To improve the model training each image has a resize to 200x200, 25-pixel padding added to each side, a random horizontal flip, a random rotation of 10 degrees, and a random perspective.

#### Train Data Transforms
```python
train_tfms = tt.Compose([
                         tt.Resize((200, 200)),
                         tt.RandomCrop(200, padding=25, padding_mode='reflect'),
                         tt.RandomHorizontalFlip(),
                         tt.RandomRotation(10),
                         tt.RandomPerspective(distortion_scale=0.2),
                         tt.ToTensor()
                        ])                       
```
The images used for testing the model were first preprocessed using a [tool](https://github.com/danielgatis/rembg) to remove the background of an image. This [background removal model](https://github.com/danielgatis/rembg) is also used in the backend to preprocess images before the evaluation using ResNet152. That allows us to avoid the background of the gestures confusing the model which increases the accuracy of classification.

An example of image preprocessing using this tool:

<br/>
<div align="center" style="display:grid; 
            justify-content: center;
            padding:15px 0 30px 0">
    <div>
        <img style="margin-right: 35px" src = "https://imgur.com/zqFGk1u.png" />
	    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <img style="margin-left: 35px" src = "https://imgur.com/1cVVKhi.png" />
    </div>
</div>
<br/>

The initial model for the website was a simple convolutional neural network built from scratch. There were several different variations of the built-from-scratch model, however, the best overall was a one-dimensional image classification model trained on images 75x75, which had 4 ReLU dense layers, and 3 ReLU convolutional layers.

#### Units per Layer
```python
self.dense_layer_1 = nn.Linear(1470, 800)
self.dense_layer_2 = nn.Linear(800, 600)
self.dense_layer_3 = nn.Linear(600, 500)
self.dense_layer_4 = nn.Linear(500, 300)
self.output_layer = nn.Linear(300, 26)
```
#### Convolutional Layer Parameters
```python
kernel_size1=5, stride1=2
kernel_size2=4, stride2=1
kernel_size3=3, stride3=1
```
#### Hyperparameters
```python
epochs = 4
criterion = torch.nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
```
But, that initial approach was incorrect, since image classification is much more complex, and such a simple model would not be able to produce good results. Even though a combination with background removal allowed us to achieve high testing and training accuracy for the initial model, it failed with the evaluation of other datasets that were not used for training. The best accuracy on the non-training dataset was 17%, which is more than a probability of random guessing (1/26 = 3.8%), but still not enough for good classification of gestures.

Because of the reason described above, a pre-trained image classification model was used. The first version was ResNet34. It immediately was able to produce a much higher accuracy on non-training datasets (60% to 70% depending on the dataset used for testing), unlike the built-from-scratch model. It was clear that the overfitting didn't happen anymore, and the usage of a pre-trained model was the right path. Different experiments were performed to make the accuracy even higher, but the best solution again was to use another more complex model, which is the ResNet152. A model similar to ResNet152, Regnet_y_32gf, was also tested, but the accuracy results were similar, so the decision was to stay with ResNet152. The parameters used to train pre-trained models are identical to the parameters used for the final model. Testing data was also first preprocessed using [background removal model](https://github.com/danielgatis/rembg).

### ML Dataset Description
#### Synthetic
Dataset used for training the built-from-scratch model. The idea was to make our model distinguish between hand and background since the synthetic dataset contains a lot of images with simulated backgrounds. It didn't work, so the background removal model was used instead.

https://www.kaggle.com/datasets/allexmendes/asl-alphabet-synthetic

#### Londhe
Dataset used at first to test the built-from-scratch model trained with a Synthetic dataset. Since the accuracy was 0%, this dataset was used after to train the built-from-scratch model, while synthetic was used only for testing. There was no improvement.

https://www.kaggle.com/datasets/kapillondhe/american-sign-language

#### Akash
Dataset used to train the ResNet34 and ResNet152. It was also used to test the built-from-scratch model. For some experiments, images with pairs of often misclassified gestures were replaced, but that didn't improve accuracy, so the default Akash dataset was used to get the final version of our model based on ResNet152.

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

#### Rasband
The primary testing dataset was used for ResNet34, ResNet152, and Regnet_y_32gf. The decision to either continue to improve the model or to stop was primarily based on the Rasband accuracy. This dataset has a good variety of hands and backgrounds. 

https://www.kaggle.com/datasets/danrasband/asl-alphabet-test

#### PHP 
Dataset used once to train ResNet34. It was used in the experiment to improve the accuracy of ResNet34.

https://www.kaggle.com/datasets/phphuc/overlay-asl-dataset

#### Geislinger
Dataset used once to test built-from-scratch model. Images didn't have a fixed size, so the dataset wasn't used further anymore.

https://www.kaggle.com/datasets/phphuc/overlay-asl-dataset

#### Yauheni
Dataset used for model testing. Created by the project contributor, Yauheni Patapau.

https://drive.google.com/file/d/1mBNmI6Vuiq4vj8Mzrpv7-i0Rs2VfEKNL/view?usp=share_link

#### Luis
Dataset used for model testing. Created by the project contributor, Luis Medina.

https://drive.google.com/file/d/1JrjblUbsrpmFybwOZ1mEJdVmdI0dVIjL/view?usp=share_link


### ML Model Experimental Performance

### User Interface Design

## Code Organization
* ASL-Learner / front end
  - public: this folder contains all the images for website logos
  - src: this folder contains all the .js, .css files. This folder also contains components, asl_letters, images folders
  	- asl_letters: this folder contains all sign images for every letter
	- components: it conatins components can be used in the website
	- images: it contains images for the homepage
	- App.css: it contains the styling for app.js file
	- App.js: this file connects all the pages and creates path for them
	- App.test.js: it tests the App.js file
	- chooseLevel.css: it contains the styling for chooseLevel.jsx
	- chooseLevel.jsx: it contains all the level and words inside each level which a user can choose
	- game.css: it contains the styling for the game.jsx file
	- game.jsx: it is the portion where a user tries to mimic fingerspelling of each letters of a word to get points
	- home.css: it contains styling for home.js file
	- home.js: it is the front page or default page that a user interacts at first
	- index.css: it contains styling for index.js file
	- index.js: it is the root file that contains App.js file to browse through other pages
	- info.css: it contains styling for info.jsx file
	- info.jsx: it contains user info where a user can access their scores and username
	- Login.js: it allows user to login to the website
	- logo.svg: it is default file from react
	- Register.css: it contains styling for Register.js file
	- Register.js: it allows new users to sign up and create a new account in the website
	- reportWebVitals.js: it is a default file from react
	- setupTests.js: it is a defualt file from react
	- webcam.css: it contains styling for webcam.js and webcam.html files
	- webcam.html: it contains the html portion of webcam
	- webcam.js: it contains .js portion of webcam
	- webcams.js: it allows the webcam to access in the website
  - package.json: it has the all the packages
  - package-lock.json: it has more hidden packages that is used in the website
  
* backend
  - _pycache_: it contains all the python caches of the project
  - instance: it has the databases of the project
  - app.py: it is the python file that is responsible for fetching the image array, transforming it and evaluating it.
  - model.py: its a python file that contains code for model class and evaluation functions
* model_training.ipynb: it is the jupiter file that trained the model
* README.md: it is the readme of the project
  
  
## How to Run

## How to Train

## Challenges Faced
### ML side
The model ResNet152 used for the project is not ideal. Some gestures (particularly gestures for pairs of letters E&S, K&V, and G&H) are often misclassified because of the similarity between them. See the example for letters E and S below:

<br/>
<div align="center" style="display:grid; 
            justify-content: center;
            padding:15px 0 30px 0">
    <div>
        <img style="margin-right: 35px" src = "https://imgur.com/SjzD6Gz.png" />
	    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <img style="margin-left: 35px" src = "https://imgur.com/RWperuv.png" />
    </div>
</div>
<br/>

We have tried to implement two possible solutions to it. One was a multiple-frame classification, and another was to use additional binary models that would classify one of the letters in those pairs.

However, it didn’t seem to improve the user experience and we’ve decided to implement custom thresholds in the backend for those letters since the purpose of our project is to teach a user how to do the fingerspelling. The human mind is much more advanced than our model. If a user places his thumb finger just a little bit differently from what our model could understand, another person would be able to identify the gesture from the context correctly.

### Frontend/Backend
One of the front-end challenges was figuring out which react hooks to use. Another challenge was for the register and login pages when we tried to use an API call to save user info or create a new user at the backend. We had to use the useState() function to make that possible. We also had to give user states for each pages to know which user it is.


## Future Work

## The Team
This project is the combined effort of ...
