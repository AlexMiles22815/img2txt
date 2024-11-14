idk what is that that just works lol.
# Instalision 
First of all install python => 3.9, then instal req.txt
```python
pip install -r req.txt
```
And then you are ready to go!

# Traning
### Dataset
To start train your own model for this ai, you need a dataset. Dataset should contains 2 dirs: images and captions.
Place your .jpg, .png, .jpeg images into images dir, and .txt captions into captions dir.
Names of files should match. For example:

```python
images/img1.png
captions/img1.txt
```
### Params
To set your own params of traning, edit values in ```train.py``` file.

# Inference
Use:
```python
python inference.py --image_path path/to/image
```
