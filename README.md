# listed_assignment_image_caption
This is an interactive Image Caption generator webapp made using streamlit. It has an option to upload an image of any type then printing out the description of the image.
The image so uploaded goes through the Vision Transformer and GPT Transforemr(ViT-gpt2 Transformer) and outputs a text describing the image.
Here, State of art HuggingFace Library has been used and the relevent cheakpoint "nlpconnect/vit-gpt2-image-captioning" has been used while compiling the Vision Transformer for Encoder and gpt2 as Decoder.
Initialization of the VisionEncoderDecoderModel, ViTFeatureExtractor and AutoTokenizer is done using the from_pretrained() function and passing the model weights context("nlpconnect/vit-gpt2-image-captioning"), thus having our pre-trained model objects ready.
At first, the input image is cheaked to be RGB, if found not, then converted into one using the PIL(pillow) library, then it is passed to the feature_extractor part where the output is an array which is basically called a tensor defining all possible key features of the raw data in numerical form.
Then by using the model object the pixel_values are passed giving the output as tensor signifying the output_ids for the converted image to text part and finally using the AutoTokenizer to decode the output_ids into the predicted text.
