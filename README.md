# Re-id-and-realness

The model is live and hosted on PythonAnywhere 

To use it 

```
POST request to "http://tirth5828.pythonanywhere.com/predict"

Body : 
{
    "image1": "link to image 1",
    "image2": "link to image 2"
}


Sample output : 
{
    "realness_score1": 0.7700245098039216,
    "realness_score2": 0.7700245098039216,
    "similarity_score": [
        1.0,
        0.06315789473684211,
        0.2545454545454545,
        0.25,
        0.06315789473684211,
        1.0,
        0.3181818181818182,
        0.1651376146788991,
        0.25,
        0.1651376146788991,
        0.3188405797101449,
        1.0,
        0.2545454545454545,
        0.3181818181818182,
        1.0,
        0.3188405797101449
    ],
    "error": []
}
```
