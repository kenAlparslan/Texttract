# TextTract Server - Client App
Front end: HTML
Backend: Python Flask
Database: None
Deployment: Docker

Check out the blog post here >> https://realpython.com/blog/python/setting-up-a-simple-ocr-server/

# Docker OSX Install
Install Docker for Mac or Docker for Windows https://hub.docker.com/editions/community/docker-ce-desktop-windows/
Build the container and run the image...

```sh
$ docker build --rm -t flask-ocr .
$ docker run -p 5000:5000 flask-ocr
```
If localhost:80 is busy, simply change it to a different port number. First 5000 is your localhost port number, 5000 is the docker's binded port number.
For more info on dockers, check here:https://github.com/docker/getting-started

## Test images
https://files.realpython.com/media/ocr.930a7baf9137.jpg
https://files.realpython.com/media/sample1.a36a230755dc.jpg
https://files.realpython.com/media/sample2.36f8074c5273.jpg
https://files.realpython.com/media/sample3.8d93cef43018.jpg
https://files.realpython.com/media/sample4.c68c31b95ffb.jpg
https://files.realpython.com/media/sample5.ca470b17f6d7.jpg

## Current Docker Size
1.25GB