if (typeof window.dash_clientside === 'undefined') {
    window.dash_clientside = {};
}

if (typeof window.dash_clientside.clientside === 'undefined') {
    window.dash_clientside.clientside = {};
}


window.dash_clientside.clientside.transform_image = function(logToggle, data) {
    console.log("Received logToggle:", logToggle); // Check logToggle value
    console.log("Received src:", data); // Check src value
    src = data;
    // If logToggle is false or src is not provided, return the original src
    if (!logToggle || !src) {
        console.log("Returning original image without transformation.");
        return Promise.resolve(src);
    }

    return new Promise(function(resolve, reject) {
        // Create an Image element
        var image = new Image();
        image.onload = function() {
            // If logToggle is true, proceed with the transformation
            var canvas = document.createElement('canvas');
            canvas.width = image.width;
            canvas.height = image.height;
            var ctx = canvas.getContext('2d');

            ctx.drawImage(image, 0, 0, image.width, image.height);
            var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            var data = imageData.data;
            var floatData = new Float32Array(data.length);

            // Apply log(1+x) transformation to each pixel
            var min = Infinity;
            var max = -Infinity;
            for (var i = 0; i < data.length; i += 4) {
                floatData[i] = Math.log1p(data[i]);       // Red
                floatData[i + 1] = Math.log1p(data[i + 1]); // Green
                floatData[i + 2] = Math.log1p(data[i + 2]); // Blue
                // Alpha channel remains unchanged

                // Update min and max
                min = Math.min(min, floatData[i], floatData[i + 1], floatData[i + 2]);
                max = Math.max(max, floatData[i], floatData[i + 1], floatData[i + 2]);
            }

            // Apply min-max normalization and scale to 0-255
            for (var i = 0; i < floatData.length; i += 4) {
                floatData[i] = (floatData[i] - min) / (max - min) * 255;       // Red
                floatData[i + 1] = (floatData[i + 1] - min) / (max - min) * 255; // Green
                floatData[i + 2] = (floatData[i + 2] - min) / (max - min) * 255; // Blue
                floatData[i + 3] = data[i + 3]; // Alpha channel remains unchanged
            }

            // Convert floatData back to Uint8ClampedArray for imageData
            for (var i = 0; i < data.length; i++) {
                data[i] = Math.round(floatData[i]);
            }

            ctx.putImageData(imageData, 0, 0);
            resolve(canvas.toDataURL()); // Convert the canvas back to a base64 URL
        };
        image.onerror = function() {
            console.error("Failed to load image");
            reject(new Error('Failed to load image'));
        };
        image.src = src;
    });
}

window.dash_clientside.clientside.transform_raw_data = function(logToggle, mask, data, minMaxValues) {
    console.log("Received logToggle:", logToggle); // Check logToggle value
    console.log("Received mask:", mask); // Check maskPath value
    console.log("Received minMaxValues:", minMaxValues); // Check minMaxValues value
    console.log("Received data:", data); // Check data value
    if (typeof data === 'undefined' || data === null || !Array.isArray(data)) {
        console.log("Data is not defined or null.");
        console.log("Returning original image without transformation.");
        return Promise.resolve(src);
    }
    src = data;
    // If src is not provided, or maskPath or minMaxValues have changed, return the original src
    if (!src || !minMaxValues || minMaxValues.length !== 2) {
        console.log("Returning original image without transformation.");
        return Promise.resolve(src);
    }
    console.log("Received data size:", data.length); // Check data value

    return new Promise(function(resolve, reject) {
        var maskImage = new Image();
        var outImage = new Image();

        if (data[0].length > data.length) {
            outImage.width = 200;
            outImage.height = 200 * (data.length / data[0].length);
        } else {
            outImage.width = 200 * (data[0].length / data.length);
            outImage.height = 200;
        }

        // Get the dimensions of the original 2D array
        var originalWidth = data[0].length;
        var originalHeight = data.length;

        var maskPromise = mask ?
            new Promise((resolve, reject) => { maskImage.onload = resolve; maskImage.onerror = reject; maskImage.src = mask; }) :
            Promise.resolve();

        Promise.all([
            maskPromise
        ]).then(() => {
            // If logToggle is true, proceed with the transformation
            var data = src;

            // Flatten the data
            data = data.flat();

            // Create a new canvas and context
            var canvas = document.createElement('canvas');
            var ctx = canvas.getContext('2d');

            // Set the canvas size to match the image size
            canvas.width = originalWidth;
            canvas.height = originalHeight;

            console.log("Canvas width:", canvas.width);
            console.log("Canvas height:", canvas.height);

            var maskData;
            if (mask) {
                console.log("Mask image loaded successfully");
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                // Resize the mask if it's larger than the image
                ctx.drawImage(maskImage, 0, 0, originalWidth, originalHeight);
                var tempMaskData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
                maskData = new Uint8Array(data.length);
                var j = 0;
                for (var i = 0; i < data.length; i++) {
                    maskData[i] = tempMaskData[j] < 254 ? 0 : tempMaskData[j];
                    j += 4;
                }
            } else {
                // Default mask of all 1s
                maskData = new Uint8Array(data.length).fill(1);
            }

            // Determine the number of channels
            var numChannels = src.numChannels || 1; // Assuming src.numChannels is defined. If not, default to 1.

            // Apply mask and log transformation to each pixel
            new_min = 255;
            new_max = 0;
            var new_data = new Float32Array(data.length / numChannels);
            console.log("New data length:", new_data.length);

            for (var i = 0; i < new_data.length; i += numChannels) {
                var grey = data[i * numChannels];
                if (maskData[i]!=0 && grey>0 && !isNaN(grey)){
                    // Clip data between min - max values
                    grey = Math.max(minMaxValues[0], Math.min(minMaxValues[1], grey));
                    // Normalize the data between the min-max values
                    grey = (grey - minMaxValues[0]) / (minMaxValues[1] - minMaxValues[0]);
                    if (logToggle) {
                        grey = Math.log(grey + 0.000000000001);
                    }
                    if (grey < new_min) {
                        new_min = grey;
                    }
                    if (grey > new_max) {
                        new_max = grey;
                    }
                    new_data[i] = grey;

                } else {
                    new_data[i] = 0;
                }

            }
            console.log("New min value:", new_min);
            console.log("New max value:", new_max);

            var reshapedData = new Uint8ClampedArray(originalWidth * originalHeight * 4);

            for (var i = 0, j = 0; i < new_data.length; i++, j += 4) {
                var tmp = Math.round((new_data[i] - new_min) / (new_max - new_min) * 255);
                if (maskData[i] != 0){
                    reshapedData[j] = tmp;     // Red channel
                    reshapedData[j + 1] = tmp; // Green channel
                    reshapedData[j + 2] = tmp; // Blue channel
                    reshapedData[j + 3] = 255; // Alpha channel
                }
                else {
                    reshapedData[j] = 0;     // Red channel
                    reshapedData[j + 1] = 0; // Green channel
                    reshapedData[j + 2] = 0; // Blue channel
                    reshapedData[j + 3] = 255; // Alpha channel
                }
            }

            // Create a new ImageData object
            var imageData = new ImageData(reshapedData, originalWidth, originalHeight);

            // Create a temporary canvas to hold the original image
            var tempCanvas = document.createElement('canvas');
            var tempCtx = tempCanvas.getContext('2d');
            tempCanvas.width = originalWidth;
            tempCanvas.height = originalHeight;
            tempCtx.putImageData(imageData, 0, 0);

            // Change the canvas dimensions to the new size
            canvas.width = outImage.width;
            canvas.height = outImage.height;

            // Clear the canvas and draw the image again, this time resizing it
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(tempCanvas, 0, 0, originalWidth, originalHeight, 0, 0, canvas.width, canvas.height);

            // Convert the canvas to a data URL
            var dataUrl = canvas.toDataURL();

            // Set the source of the outImage to the data URL of the canvas
            outImage.src = dataUrl;
            resolve(dataUrl); // Convert the canvas back to a base64 URL

        }).catch((err) => {
            console.error("Failed to load image", err);
            reject(new Error('Failed to load image'));
        });
    });
}

window.dash_clientside.clientside.expanded_transform_image = function(logToggle, mask, minMaxValues, data) {
    console.log("Received logToggle:", logToggle); // Check logToggle value
    console.log("Received mask:", mask); // Check maskPath value
    console.log("Received minMaxValues:", minMaxValues); // Check minMaxValues value
    src = data;
    // // If src is not provided, or maskPath or minMaxValues have changed, return the original src
    if (!src || !minMaxValues || minMaxValues.length !== 2) {
        console.log("Returning original image without transformation.");
        return Promise.resolve(src);
    }

    return new Promise(function(resolve, reject) {
        // Create an Image element
        var image = new Image();
        var maskImage = new Image();

        var maskPromise = mask ?
            new Promise((resolve, reject) => { maskImage.onload = resolve; maskImage.onerror = reject; maskImage.src = mask; }) :
            Promise.resolve();

        Promise.all([
            new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = src; }),
            maskPromise
        ]).then(() => {
            // If logToggle is true, proceed with the transformation
            var canvas = document.createElement('canvas');
            canvas.width = image.width;
            canvas.height = image.height;
            var ctx = canvas.getContext('2d');

            ctx.drawImage(image, 0, 0, image.width, image.height);
            var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            var data = imageData.data;

            var maskData;
            if (mask) {
                console.log("Mask image loaded successfully");
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                // Resize the mask if it's larger than the image
                ctx.drawImage(maskImage, 0, 0, image.width, image.height);
                var tempMaskData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
                maskData = new Uint8Array(tempMaskData.length);
                for (var i = 0; i < tempMaskData.length; i++) {
                    maskData[i] = tempMaskData[i] < 254 ? 0 : tempMaskData[i];
                }
            } else {
                console.log("Loading default mask");
                // Default mask of all 1s
                maskData = new Uint8Array(data.length / 1).fill(1);
            }
            // console.log("Min value of mask:", Math.min(...maskData));
            // console.log("Max value of mask:", Math.max(...maskData));

            // Apply mask and log transformation to each pixel
            new_min = 255;
            new_max = 0;
            var new_data = new Float32Array(data.length / 1);

            for (var i = 0; i < data.length; i += 4) {
                var red = data[i];
                var green = data[i + 1];
                var blue = data[i + 2];

                if (maskData[i]!=0 && red>0 && green>0 && blue>0 && !isNaN(red) && !isNaN(green) && !isNaN(blue)){

                    // Clip data between min - max values
                    red = Math.max(minMaxValues[0], Math.min(minMaxValues[1], red));
                    green = Math.max(minMaxValues[0], Math.min(minMaxValues[1], green));
                    blue = Math.max(minMaxValues[0], Math.min(minMaxValues[1], blue));

                    // Normalize the data between the min-max values
                    red = (red - minMaxValues[0]) / (minMaxValues[1] - minMaxValues[0]);
                    green = (green - minMaxValues[0]) / (minMaxValues[1] - minMaxValues[0]);
                    blue = (blue - minMaxValues[0]) / (minMaxValues[1] - minMaxValues[0]);

                    if (logToggle) {
                        red = Math.log(red + 0.000000000001);
                        green = Math.log(green + 0.000000000001);
                        blue = Math.log(blue + 0.000000000001);
                    }

                    min_value = Math.min(red, green, blue);
                    if (min_value < new_min) {
                        new_min = min_value;
                    }
                    max_value = Math.max(red, green, blue);
                    if (max_value > new_max) {
                        new_max = max_value;
                    }

                    new_data[i] = red;
                    new_data[i + 1] = green;
                    new_data[i + 2] = blue;

                } else {

                    new_data[i] = 0;
                    new_data[i + 1] = 0;
                    new_data[i + 2] = 0;

                }

            }

            for (var i = 0; i < data.length; i += 4) {
                var red = data[i];
                var green = data[i + 1];
                var blue = data[i + 2];

                if (maskData[i] != 0 && red>0 && green>0 && blue>0 && !isNaN(red) && !isNaN(green) && !isNaN(blue)){
                    data[i] = Math.round((new_data[i] - new_min) / (new_max - new_min) * 255);
                    data[i + 1] = Math.round((new_data[i + 1] - new_min) / (new_max - new_min) * 255);
                    data[i + 2] = Math.round((new_data[i + 2] - new_min) / (new_max - new_min) * 255);
                }
            }

            ctx.putImageData(imageData, 0, 0);
            resolve(canvas.toDataURL()); // Convert the canvas back to a base64 URL
        }).catch((err) => {
            console.error("Failed to load image", err);
            reject(new Error('Failed to load image'));
        });
    });
}
