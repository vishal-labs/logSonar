# logSonar

A new way of visualising logs

Generally, when we want to visualise logs from a distributed system, we use grafana/prometheus and get some metrics and understanding of what is happening to the system and have some sort of filtering and sorting to it. But i felt that those charts and graphs never gave me a quick "feel" of performance of any particular node.

### My solution:

Most of the application that are setup in a distributed way already have some sort of log collection techniques for example Otel Collectors..
Now My application runs on top of this Otel and visualised it in a different way.

### How does the visualisation process look like?

- As an example, here right now i have a flask application(toy application) which generates sample logs with few parameters like service, timestamp, latency, message.
- Now, the actual logic is in the processLogic which takes in the emitted json from the above application and processes it. if i simplify the explanation of prcoessing, it basically gives different weights to each parameter and normalises it to a value between 0 - 255 and 3 of them(if you felt something, you are right!)
- now the normalised values have the essense of the response of the JSON response and we could say that it got "translated" into something new. - The Interesting thing is that, these 3 values can be represented as (R, G, B) and this will represent a pixel.
- Now my planning is that, we could get a batch of logs and get the pixels filled up WRT to a particular service(check the bellow sample image) and then we can have an understanding, if there are any repetition of issues of any particular service or node.

![Sample Output Image](./images/image.png)
