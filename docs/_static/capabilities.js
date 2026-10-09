// Collapse the code-and-test line under each capability behind one toggle.
document.addEventListener("DOMContentLoaded", () => {
  for (const list of document.querySelectorAll(".capabilities")) {
    list.classList.add("collapsed");
    const button = document.createElement("button");
    button.className = "capabilities-toggle";
    button.type = "button";
    button.textContent = "Show code and tests";
    button.addEventListener("click", () => {
      const collapsed = list.classList.toggle("collapsed");
      button.textContent = collapsed ? "Show code and tests" : "Hide code and tests";
    });
    list.before(button);
  }
});
