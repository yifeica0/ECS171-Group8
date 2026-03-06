function renderText(words, container){
    /*
    creates HTML with sentiment classes
    This function takes in a list of words with their corresponding sentiment labels and renders the text with different colors based on the sentiment.
    */

    let html = ""

    words.forEach(w => {
        if (w.sentiment === "positive") {
            html += `<span class="positive">${w.word}</span> `
        }
        else if (w.sentiment === "negative") {
            html += `<span class="negative">${w.word}</span> `
        }
        else {
            html += `<span class="neutral">${w.word}</span> `
        }
    })

    container.innerHTML = html
}