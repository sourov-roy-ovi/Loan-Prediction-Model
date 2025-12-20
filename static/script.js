function validateForm() {
    const inputs = document.querySelectorAll("input");
    for (let i of inputs) {
        if (i.value === "" || i.value < 0) {
            alert("Please enter valid inputs");
            return false;
        }
    }
    return true;
}