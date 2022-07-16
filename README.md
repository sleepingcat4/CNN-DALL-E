## What’s CNN?

Within Deep Learning, a Convolutional Neural Network or CNN is a type of

artiﬁcial neural network, which is widely used for image/object recognition

and classiﬁcation. Deep Learning thus recognizes objects in an image by using

a CNN.

## What’s DALL-E?

DALL-E and DALL-E 2 are transformer models developed by OpenAI to generate

digital images from natural language descriptions. Its name is a portmanteau of

WALL-E and Salvador Dalí.

## Project Objective

The project focused on proving that DALL-E generated images can be used as

high-quality Synthetic image datasets for training Artiﬁcial Intelligence for Image

Classiﬁcation tasks. With that objective/hypothesis in mind, our team at alone!

ai trained a CNN oriented AI model using Pytorch. All the images for the AI

model were taken from DALL-E.

## Reasons to do so?

Recently from the emergence of text-to-Image generating models such as

DALL-E 2 and Google Imagen, we have found out that these models can be used

for generating image datasets of choice. FInding quality Image Datasets for

training AI models can be hard and expensive. Not everyone has the resources

to aﬀord costly high-quality image datasets, mainly when you’re a student

researcher. Our researcher hypothesised by looking at high-quality images from

text-to-image models as DALL-E that these models enable a cheap way to train

AI models with any sort of Image datasets of your choice.

To conﬁrm our hunch, we created a simple CNN model using Pytorch to train it

on DALL-E generated Oil Paintings and Pencil Drawings images. Successfully our

model was able to successfully train itself from DALL-E generated image

datasets.

DALL-E is still in beta mode. Only few researchers and artists have access to it.

We hope that, once the API is made public, OpenAI will look into the prospect of

DALL-E generated high-quality image datasets.

## Outcomes

From our model that was trained on 15 Oil painting images and 15 Pencil

Painting images we were able to attain a 30% training accuracy and 28% testing

accuracy on average. Given to our image dataset, which is a good ratio as we

consider.

## Drawbacks

Since we had tried to prove our hypothesis, we didn’t use a large dataset rather

a small one. Considering the outcomes we’re sure that, with more DALL-E

images, about 125 images per label, a CNN model can attain 100% accuracy

from DALL-E generated image datasets.

## Interesting points

We propose further investigation in GPT-3 for its ability to create Synthetic data.

And we’re curious how OpenAI could handle people who’re generating datasets

from who aren’t.

Besides that, It’s possible to train a model using one single image with just slight

variations using the variable option. I have trained my model that way to ﬁnd

interesting outcomes.

## Yay facts

Further we have deployed a readily usable web-app created using the DALL-E

generated image dataset same as this project to deploy an image classiﬁer using

lobe.ai using Microsoft Azure. Check it out! http://dalle.azurewebsites.net/

(Python 3.8 has been used)
