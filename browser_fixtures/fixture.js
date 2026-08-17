(() => {
  const svg = color => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="180" height="120"><rect width="180" height="120" fill="${color}"/></svg>`)}`;
  const red = svg("#d7191c");
  const blue = svg("#2c7bb6");
  document.documentElement.style.setProperty("--red", `url("${red}")`);
  document.documentElement.style.setProperty("--blue", `url("${blue}")`);
  document.querySelector("#static-img").src = red;
  document.querySelector("#wide-source").srcset = blue;
  document.querySelector("#responsive-img").src = red;
  document.querySelector("#lazy-img").src = blue;
  document.querySelector("#rapid-img").src = red;
  document.querySelector("#spoof-badge").src = blue;

  const context = document.querySelector("#canvas-2d").getContext("2d");
  context.fillStyle = "#2c7bb6";
  context.fillRect(0, 0, 180, 120);

  const addShadow = (host, mode, source) => {
    const root = host.attachShadow({mode});
    const image = document.createElement("img");
    image.src = source;
    image.width = 180;
    image.height = 120;
    root.append(image);
  };
  addShadow(document.querySelector("#open-shadow"), "open", blue);
  addShadow(document.querySelector("#closed-shadow"), "closed", red);

  const blob = new Blob([`<svg xmlns="http://www.w3.org/2000/svg" width="180" height="120"><rect width="180" height="120" fill="#2c7bb6"/></svg>`], {type: "image/svg+xml"});
  const blobUrl = URL.createObjectURL(blob);
  const blobImage = document.querySelector("#blob-img");
  blobImage.addEventListener("load", () => URL.revokeObjectURL(blobUrl), {once: true});
  blobImage.src = blobUrl;

  const state = window.__AIBLINK_FIXTURES__ = {generation: 0, sentinel: "red", raceComplete: false, stormNodes: 0};
  document.querySelector("#race").addEventListener("click", () => {
    const image = document.querySelector("#rapid-img");
    let remaining = 20;
    state.raceComplete = false;
    const timer = setInterval(() => {
      state.generation += 1;
      state.sentinel = state.generation % 2 ? "blue" : "red";
      image.src = state.sentinel === "blue" ? blue : red;
      image.dataset.generation = String(state.generation);
      remaining -= 1;
      if (!remaining) {
        clearInterval(timer);
        state.raceComplete = true;
        document.querySelector("#status").value = `race complete: generation=${state.generation} sentinel=${state.sentinel}`;
      }
    }, 100);
  });
  document.querySelector("#storm").addEventListener("click", () => {
    const container = document.createElement("div");
    container.hidden = true;
    document.body.append(container);
    let batches = 0;
    const timer = setInterval(() => {
      const fragment = document.createDocumentFragment();
      for (let index = 0; index < 200; index += 1) {
        const image = document.createElement("img");
        image.src = index % 2 ? red : blue;
        fragment.append(image);
      }
      container.replaceChildren(fragment);
      state.stormNodes += 200;
      batches += 1;
      if (batches === 20) {
        clearInterval(timer);
        container.remove();
        document.querySelector("#status").value = `storm complete: mutations=${state.stormNodes}`;
      }
    }, 25);
  });
})();
