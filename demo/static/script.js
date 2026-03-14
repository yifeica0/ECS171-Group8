function renderText(words, container) {
    container.innerHTML = "";

    words.forEach(item => {
        const span = document.createElement("span");
        span.textContent = item.word + " ";
        span.style.display = "inline-block";
        span.style.margin = "6px 6px 6px 0";
        span.style.padding = "8px 14px";
        span.style.borderRadius = "14px";
        span.style.fontSize = "16px";
        span.style.border = "1px solid #d0d4dc";
        span.style.transition = "0.2s ease";

        if (item.sentiment === "good") {
            span.style.backgroundColor = "rgba(25, 135, 84, 0.28)";
            span.style.color = "#146c43";
            span.style.fontWeight = "700";
        } 
        else if (item.sentiment === "bad") {
            span.style.backgroundColor = "rgba(220, 53, 69, 0.25)";
            span.style.color = "#b02a37";
            span.style.fontWeight = "700";
        } 
        else {
            span.style.backgroundColor = "rgba(108, 117, 125, 0.10)";
            span.style.color = "#495057";
        }

        container.appendChild(span);
    });
}