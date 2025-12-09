// Small fully-connected neural network visualizer
// Architecture: 2 inputs -> H hidden neurons -> 1 output (all sigmoid).
// H (the number of hidden neurons) can be changed with a slider.

const svg = document.getElementById("network-svg");
const stepTitleEl = document.getElementById("step-title");
const equationsEl = document.getElementById("equations");
const stepExplainerEl = document.getElementById("step-explainer");
const valuesTableEl = document.getElementById("values-table");

const inputX1 = document.getElementById("input-x1");
const inputX2 = document.getElementById("input-x2");
const targetY = document.getElementById("target-y");
const lrSlider = document.getElementById("learning-rate");
const hiddenCountSlider = document.getElementById("hidden-count");

const inputX1Val = document.getElementById("input-x1-value");
const inputX2Val = document.getElementById("input-x2-value");
const targetYVal = document.getElementById("target-y-value");
const lrVal = document.getElementById("learning-rate-value");
const hiddenCountVal = document.getElementById("hidden-count-value");

const btnPrev = document.getElementById("btn-prev");
const btnNext = document.getElementById("btn-next");
const btnReset = document.getElementById("btn-reset");

const stepItems = Array.from(document.querySelectorAll(".step-item"));

let currentStep = 0;

// Architecture and parameters
let architecture = {
  inputSize: 2,
  hiddenSize: 2,
  outputSize: 1,
};

// Weights and biases are stored in arrays for clarity:
// - w_ih[j][i]: weight from input i -> hidden neuron j
// - b_h[j]: bias for hidden neuron j
// - w_ho[j]: weight from hidden neuron j -> output
// - b_o: bias for the output neuron
let params = {};
let cache = {}; // forward and backward values for current example

// To avoid lag when dragging sliders, we batch expensive re-renders
// so they happen at most once per animation frame.
let pendingFrame = null;

function requestRender() {
  if (pendingFrame != null) return;
  pendingFrame = requestAnimationFrame(() => {
    pendingFrame = null;
    renderStep();
  });
}

function randomWeight() {
  return (Math.random() * 2 - 1) * 0.8; // in [-0.8, 0.8]
}

function initParams() {
  const H = architecture.hiddenSize;
  const I = architecture.inputSize;

  const w_ih = Array.from({ length: H }, () =>
    Array.from({ length: I }, () => randomWeight())
  );
  const b_h = Array.from({ length: H }, () => randomWeight());

  const w_ho = Array.from({ length: H }, () => randomWeight());
  const b_o = randomWeight();

  params = { w_ih, b_h, w_ho, b_o };
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidPrime(x) {
  const s = sigmoid(x);
  return s * (1 - s);
}

function fmt(x, digits = 4) {
  if (typeof x === "string") return x;
  if (Number.isFinite(x)) return x.toFixed(digits);
  return String(x);
}

function computeForward() {
  const x = [parseFloat(inputX1.value), parseFloat(inputX2.value)];
  const yTrue = parseFloat(targetY.value);

  const H = architecture.hiddenSize;
  const z_h = new Array(H);
  const h = new Array(H);

  for (let j = 0; j < H; j++) {
    let sum = 0;
    for (let i = 0; i < architecture.inputSize; i++) {
      sum += params.w_ih[j][i] * x[i];
    }
    sum += params.b_h[j];
    z_h[j] = sum;
    h[j] = sigmoid(sum);
  }

  let z_o = 0;
  for (let j = 0; j < H; j++) {
    z_o += params.w_ho[j] * h[j];
  }
  z_o += params.b_o;
  const yPred = sigmoid(z_o);

  const loss = 0.5 * Math.pow(yPred - yTrue, 2); // squared error

  cache = { x, yTrue, z_h, h, z_o, yPred, loss };
}

function computeBackward() {
  const { x, yTrue, z_h, h, z_o, yPred } = cache;
  const H = architecture.hiddenSize;

  const dL_dy = yPred - yTrue;
  const dy_dz_o = sigmoidPrime(z_o);
  const dL_dz_o = dL_dy * dy_dz_o;

  const dL_dw_ho = new Array(H);
  const dL_dh = new Array(H);
  const dh_dz_h = new Array(H);
  const dL_dz_h = new Array(H);
  const dL_dw_ih = Array.from({ length: H }, () => new Array(architecture.inputSize));
  const dL_db_h = new Array(H);

  for (let j = 0; j < H; j++) {
    dL_dw_ho[j] = dL_dz_o * h[j];
    dL_dh[j] = dL_dz_o * params.w_ho[j];
    dh_dz_h[j] = sigmoidPrime(z_h[j]);
    dL_dz_h[j] = dL_dh[j] * dh_dz_h[j];

    for (let i = 0; i < architecture.inputSize; i++) {
      dL_dw_ih[j][i] = dL_dz_h[j] * x[i];
    }
    dL_db_h[j] = dL_dz_h[j];
  }

  const dL_db_o = dL_dz_o;

  Object.assign(cache, {
    dL_dy,
    dy_dz_o,
    dL_dz_o,
    dL_dw_ho,
    dL_db_o,
    dL_dh,
    dh_dz_h,
    dL_dz_h,
    dL_dw_ih,
    dL_db_h,
  });
}

function applyGradientDescent() {
  const lr = parseFloat(lrSlider.value);
  const H = architecture.hiddenSize;

  const { dL_dw_ih, dL_db_h, dL_dw_ho, dL_db_o } = cache;

  for (let j = 0; j < H; j++) {
    for (let i = 0; i < architecture.inputSize; i++) {
      params.w_ih[j][i] -= lr * dL_dw_ih[j][i];
    }
    params.b_h[j] -= lr * dL_db_h[j];
    params.w_ho[j] -= lr * dL_dw_ho[j];
  }
  params.b_o -= lr * dL_db_o;
}

// SVG drawing
const layout = {
  input: [],
  hidden: [],
  output: [],
};

function updateLayout() {
  layout.input = [
    { id: "x1", x: 80, y: 80 },
    { id: "x2", x: 80, y: 220 },
  ];

  const H = architecture.hiddenSize;
  const xHidden = 280;
  const top = 60;
  const bottom = 240;
  const hidden = [];

  if (H === 1) {
    hidden.push({ id: "h1", x: xHidden, y: 150 });
  } else {
    const gap = (bottom - top) / (H - 1);
    for (let j = 0; j < H; j++) {
      hidden.push({ id: `h${j + 1}`, x: xHidden, y: top + gap * j });
    }
  }
  layout.hidden = hidden;

  layout.output = [{ id: "y", x: 480, y: 150 }];
}

function clearSvg() {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}

function createLine(x1, y1, x2, y2, cls) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("class", cls);
  return line;
}

function createCircle(x, y, r, cls) {
  const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  c.setAttribute("cx", x);
  c.setAttribute("cy", y);
  c.setAttribute("r", r);
  c.setAttribute("class", cls);
  return c;
}

function createText(x, y, text, cls) {
  const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
  t.setAttribute("x", x);
  t.setAttribute("y", y);
  t.setAttribute("class", cls);
  t.textContent = text;
  return t;
}

function drawNetwork() {
  clearSvg();

  svg.innerHTML = `
    <defs>
      <style>
        .conn { stroke: rgba(148, 163, 184, 0.8); stroke-width: 2; }
        .conn-gradient { stroke: #a855f7; stroke-width: 3; }
        .neuron { fill: #020617; stroke-width: 2; }
        .neuron-input { stroke: #22c55e; }
        .neuron-hidden { stroke: #eab308; }
        .neuron-output { stroke: #f97316; }
        .neuron-highlight { filter: drop-shadow(0 0 4px rgba(129, 140, 248, 0.9)); }
        .neuron-text { fill: #e5e7eb; font-size: 12px; text-anchor: middle; dominant-baseline: middle; }
        .label-text { fill: #9ca3af; font-size: 11px; text-anchor: middle; }
      </style>
    </defs>
  `;

  // Connections input -> hidden
  layout.input.forEach((iNode) => {
    layout.hidden.forEach((hNode) => {
      const line = createLine(iNode.x + 22, iNode.y, hNode.x - 22, hNode.y, "conn");
      line.dataset.from = iNode.id;
      line.dataset.to = hNode.id;
      svg.appendChild(line);
    });
  });

  // Connections hidden -> output
  layout.hidden.forEach((hNode) => {
    const yNode = layout.output[0];
    const line = createLine(hNode.x + 22, hNode.y, yNode.x - 22, yNode.y, "conn");
    line.dataset.from = hNode.id;
    line.dataset.to = yNode.id;
    svg.appendChild(line);
  });

  // Neurons
  layout.input.forEach((node) => {
    const c = createCircle(node.x, node.y, 18, "neuron neuron-input");
    c.dataset.id = node.id;
    svg.appendChild(c);
    svg.appendChild(createText(node.x, node.y, node.id.toUpperCase(), "neuron-text"));
  });

  layout.hidden.forEach((node) => {
    const c = createCircle(node.x, node.y, 18, "neuron neuron-hidden");
    c.dataset.id = node.id;
    svg.appendChild(c);
    svg.appendChild(createText(node.x, node.y, node.id.toUpperCase(), "neuron-text"));
  });

  layout.output.forEach((node) => {
    const c = createCircle(node.x, node.y, 20, "neuron neuron-output");
    c.dataset.id = node.id;
    svg.appendChild(c);
    svg.appendChild(createText(node.x, node.y, "ŷ", "neuron-text"));
  });

  // Labels for layers
  svg.appendChild(createText(80, 300, "Input layer", "label-text"));
  svg.appendChild(createText(280, 300, "Hidden layer", "label-text"));
  svg.appendChild(createText(480, 300, "Output layer", "label-text"));
}

function highlightForStep(step) {
  // Reset all lines & neurons
  Array.from(svg.querySelectorAll(".conn")).forEach((line) => {
    line.setAttribute("class", "conn");
  });
  Array.from(svg.querySelectorAll(".neuron")).forEach((n) => {
    n.classList.remove("neuron-highlight");
  });

  const setConnGradient = (from, to) => {
    Array.from(svg.querySelectorAll("line")).forEach((line) => {
      if (line.dataset.from === from && line.dataset.to === to) {
        line.setAttribute("class", "conn conn-gradient");
      }
    });
  };

  const highlightNeuron = (id) => {
    Array.from(svg.querySelectorAll(".neuron")).forEach((n) => {
      if (n.dataset.id === id) n.classList.add("neuron-highlight");
    });
  };

  if (step === 0) {
    // Forward pass: highlight all neurons
    ["x1", "x2", "y"].forEach(highlightNeuron);
    layout.hidden.forEach((h) => highlightNeuron(h.id));
  } else if (step === 1) {
    // Loss: just the output
    highlightNeuron("y");
  } else if (step === 2) {
    // Gradients at output: highlight output and connections from hidden
    highlightNeuron("y");
    layout.hidden.forEach((h) => {
      highlightNeuron(h.id);
      setConnGradient(h.id, "y");
    });
  } else if (step === 3) {
    // Gradients in hidden layer: highlight inputs and hidden + their connections
    ["x1", "x2"].forEach(highlightNeuron);
    layout.hidden.forEach((h) => {
      highlightNeuron(h.id);
      ["x1", "x2"].forEach((inp) => setConnGradient(inp, h.id));
    });
  } else if (step === 4) {
    // Weight update: highlight everything again
    ["x1", "x2", "y"].forEach(highlightNeuron);
    layout.hidden.forEach((h) => highlightNeuron(h.id));
  }
}

function getWeightForConnection(from, to) {
  const H = architecture.hiddenSize;

  if ((from === "x1" || from === "x2") && to && to.startsWith("h")) {
    const j = parseInt(to.slice(1), 10) - 1;
    if (j < 0 || j >= H) return null;
    const i = from === "x1" ? 0 : 1;
    return params.w_ih[j][i];
  }

  if (from && from.startsWith("h") && to === "y") {
    const j = parseInt(from.slice(1), 10) - 1;
    if (j < 0 || j >= H) return null;
    return params.w_ho[j];
  }

  return null;
}

function updateConnectionStyles(updatedSet) {
  const H = architecture.hiddenSize;
  let maxAbs = 0;

  for (let j = 0; j < H; j++) {
    for (let i = 0; i < architecture.inputSize; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(params.w_ih[j][i]));
    }
    maxAbs = Math.max(maxAbs, Math.abs(params.w_ho[j]));
  }

  const lines = Array.from(svg.querySelectorAll(".conn"));
  lines.forEach((line) => {
    const from = line.dataset.from;
    const to = line.dataset.to;
    const w = getWeightForConnection(from, to);
    if (w == null) return;

    const abs = Math.abs(w);
    const norm = maxAbs > 0 ? abs / maxAbs : 0;
    let width = 1.5 + 3 * norm;

    if (updatedSet && updatedSet.has(`${from}->${to}`)) {
      width += 1.2;
    }

    line.style.strokeWidth = String(width);
    line.style.stroke = w >= 0 ? "#60a5fa" : "#fb7185";
  });
}

// UI rendering for equations & values
function renderStep() {
  computeForward();
  computeBackward();
  highlightForStep(currentStep);

  stepItems.forEach((item, idx) => {
    item.classList.toggle("active", idx === currentStep);
  });

  const s = cache;
  const H = architecture.hiddenSize;
  let updatedConnections = null;

  if (currentStep === 0) {
    stepTitleEl.textContent = "1. Forward pass: from inputs to prediction";
    equationsEl.textContent = [
      "Step 1: Each hidden neuron j does:",
      "  z_h[j] = w_ih[j] · x + b_h[j]  (weighted sum)",
      "  h[j]   = σ(z_h[j])             (apply sigmoid)",
      "",
      "Step 2: The output neuron does:",
      "  z_out = Σ_j w_ho[j] · h[j] + b_o",
      "  ŷ     = σ(z_out)",
    ].join("\n");

    stepExplainerEl.textContent =
      "We start by mixing the two inputs inside each hidden neuron, " +
      "squashing them with a sigmoid, then mixing those hidden activations " +
      "again to get the final prediction ŷ. On the picture, thicker, brighter " +
      "lines mean larger weights.";

    const pairs = [
      ["x₁", s.x[0]],
      ["x₂", s.x[1]],
    ];
    for (let j = 0; j < H; j++) {
      pairs.push([`z_h${j + 1}`, s.z_h[j]]);
      pairs.push([`h${j + 1}`, s.h[j]]);
    }
    pairs.push(["z_out", s.z_o]);
    pairs.push(["ŷ (prediction)", s.yPred]);
    renderValues(pairs);
  } else if (currentStep === 1) {
    stepTitleEl.textContent = "2. Loss: how far is ŷ from y?";
    equationsEl.textContent = [
      "We measure error with squared loss:",
      "  L = ½ · (ŷ − y)²",
      "",
      "If ŷ is close to y, L is small.",
    ].join("\n");

    stepExplainerEl.textContent =
      "Here we simply compare the prediction ŷ to the target y. " +
      "The further apart they are, the bigger the loss.";

    renderValues([
      ["ŷ (prediction)", s.yPred],
      ["y (target)", s.yTrue],
      ["L (loss)", s.loss],
    ]);
  } else if (currentStep === 2) {
    stepTitleEl.textContent = "3. Backward: gradients at the output neuron";
    equationsEl.textContent = [
      "We move one step back from the loss to z_out:",
      "  dL/dŷ   = ŷ − y",
      "  dŷ/dz   = σ'(z_out) = σ(z_out)·(1 − σ(z_out))",
      "  dL/dz   = dL/dŷ · dŷ/dz",
      "",
      "For each hidden neuron j:",
      "  dL/dw_ho[j] = dL/dz · h[j]",
      "  dL/db_o     = dL/dz · 1",
    ].join("\n");

    stepExplainerEl.textContent =
      "We ask: if we nudge the output weights and bias a little, " +
      "how does the loss change? These are the gradients for the last layer.";

    const pairs = [
      ["dL/dŷ", s.dL_dy],
      ["σ'(z_out)", s.dy_dz_o],
      ["dL/dz_out", s.dL_dz_o],
    ];
    for (let j = 0; j < H; j++) {
      pairs.push([`dL/dw_ho${j + 1}`, s.dL_dw_ho[j]]);
    }
    pairs.push(["dL/db_o", s.dL_db_o]);
    renderValues(pairs);
  } else if (currentStep === 3) {
    stepTitleEl.textContent = "4. Backward: gradients for hidden neurons and input weights";
    equationsEl.textContent = [
      "For each hidden neuron j we go one step back:",
      "  dL/dh[j]   = dL/dz_out · w_ho[j]",
      "  σ'(z_h[j]) = σ(z_h[j])·(1 − σ(z_h[j]))",
      "  dL/dz_h[j] = dL/dh[j] · σ'(z_h[j])",
      "",
      "Then for each input i:",
      "  dL/dw_ih[j,i] = dL/dz_h[j] · x[i]",
      "  dL/db_h[j]    = dL/dz_h[j] · 1",
    ].join("\n");

    stepExplainerEl.textContent =
      "Now we push the error signal one step earlier, into each hidden neuron " +
      "and all input-to-hidden weights. This tells us how to tweak the 'feature " +
      "detectors' in the hidden layer.";

    const pairs = [];
    for (let j = 0; j < H; j++) {
      pairs.push([`dL/dh${j + 1}`, s.dL_dh[j]]);
      pairs.push([`σ'(z_h${j + 1})`, s.dh_dz_h[j]]);
      pairs.push([`dL/dz_h${j + 1}`, s.dL_dz_h[j]]);
    }
    renderValues(pairs);
  } else if (currentStep === 4) {
    const oldLoss = s.loss;

    const oldParams = {
      w_ih: params.w_ih.map((row) => row.slice()),
      b_h: params.b_h.slice(),
      w_ho: params.w_ho.slice(),
      b_o: params.b_o,
    };

    const gradSnapshot = {
      dL_dw_ih: s.dL_dw_ih,
      dL_db_h: s.dL_db_h,
      dL_dw_ho: s.dL_dw_ho,
      dL_db_o: s.dL_db_o,
    };

    applyGradientDescent();
    computeForward();
    const newLoss = cache.loss;

    stepTitleEl.textContent = "5. Update: take one gradient descent step";
    equationsEl.textContent = [
      "Gradient descent update rule (for any weight w):",
      "  w_new = w_old − α · dL/dw",
      "",
      "Changing the number of hidden neurons changes how many weights",
      "we update, but the idea is always the same.",
    ].join("\n");

    stepExplainerEl.textContent =
      "Finally, we move every weight and bias a small step in the direction " +
      "that reduces the loss. The cards below show old → new values for each " +
      "weight/bias, and the lines that change the most get a little thicker. " +
      "Repeating this step many times is what training is.";

    const v = [];
    v.push(["Old loss", oldLoss]);
    v.push(["New loss", newLoss]);

    const changed = [];

    for (let j = 0; j < H; j++) {
      for (let i = 0; i < architecture.inputSize; i++) {
        const fromLabel = i === 0 ? "x₁" : "x₂";
        const wLabel = `w(${fromLabel}→h${j + 1})`;
        const oldW = oldParams.w_ih[j][i];
        const newW = params.w_ih[j][i];
        const g = gradSnapshot.dL_dw_ih[j][i];
        v.push([
          wLabel,
          `${fmt(oldW)} → ${fmt(newW)}  (grad ${fmt(g, 3)})`,
        ]);

        const fromId = i === 0 ? "x1" : "x2";
        const toId = `h${j + 1}`;
        const deltaAbs = Math.abs(newW - oldW);
        changed.push({ from: fromId, to: toId, deltaAbs });
      }

      const oldBh = oldParams.b_h[j];
      const newBh = params.b_h[j];
      const gb = gradSnapshot.dL_db_h[j];
      v.push([
        `b(h${j + 1})`,
        `${fmt(oldBh)} → ${fmt(newBh)}  (grad ${fmt(gb, 3)})`,
      ]);
    }

    for (let j = 0; j < H; j++) {
      const oldWh = oldParams.w_ho[j];
      const newWh = params.w_ho[j];
      const g = gradSnapshot.dL_dw_ho[j];
      v.push([
        `w(h${j + 1}→ŷ)`,
        `${fmt(oldWh)} → ${fmt(newWh)}  (grad ${fmt(g, 3)})`,
      ]);

      const fromId = `h${j + 1}`;
      const toId = "y";
      const deltaAbs = Math.abs(newWh - oldWh);
      changed.push({ from: fromId, to: toId, deltaAbs });
    }

    const oldBo = oldParams.b_o;
    const newBo = params.b_o;
    const gbO = gradSnapshot.dL_db_o;
    v.push([
      "b(ŷ)",
      `${fmt(oldBo)} → ${fmt(newBo)}  (grad ${fmt(gbO, 3)})`,
    ]);

    changed.sort((a, b) => b.deltaAbs - a.deltaAbs);
    const topChanged = changed.slice(0, 4);
    updatedConnections = new Set(
      topChanged.map((c) => `${c.from}->${c.to}`)
    );

    renderValues(v, ["Old loss", "New loss"]);
  }

  updateConnectionStyles(updatedConnections);
}
}

function renderValues(pairs, highlightLabels = []) {
  valuesTableEl.innerHTML = "";
  pairs.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "value-card";
    if (highlightLabels.includes(label)) card.classList.add("highlight");

    const labelEl = document.createElement("div");
    labelEl.className = "label";
    labelEl.textContent = label;

    const valEl = document.createElement("div");
    valEl.className = "val";
    valEl.textContent = fmt(value);

    card.appendChild(labelEl);
    card.appendChild(valEl);
    valuesTableEl.appendChild(card);
  });
}

// Event wiring
[inputX1, inputX2, targetY].forEach((slider) => {
  slider.addEventListener("input", () => {
    inputX1Val.textContent = parseFloat(inputX1.value).toFixed(2);
    inputX2Val.textContent = parseFloat(inputX2.value).toFixed(2);
    targetYVal.textContent = parseFloat(targetY.value).toFixed(2);
    // Batch expensive recomputation to keep dragging smooth
    requestRender();
  });
});

lrSlider.addEventListener("input", () => {
  lrVal.textContent = parseFloat(lrSlider.value).toFixed(2);
});

hiddenCountSlider.addEventListener("input", () => {
  const H = parseInt(hiddenCountSlider.value, 10);
  architecture.hiddenSize = H;
  hiddenCountVal.textContent = String(H);
  updateLayout();
  drawNetwork();
  initParams();
  requestRender();
});

btnPrev.addEventListener("click", () => {
  currentStep = Math.max(0, currentStep - 1);
  renderStep();
});

btnNext.addEventListener("click", () => {
  currentStep = Math.min(4, currentStep + 1);
  renderStep();
});

btnReset.addEventListener("click", () => {
  initParams();
  currentStep = 0;
  renderStep();
});

// Initialize
function initUI() {
  inputX1Val.textContent = parseFloat(inputX1.value).toFixed(2);
  inputX2Val.textContent = parseFloat(inputX2.value).toFixed(2);
  targetYVal.textContent = parseFloat(targetY.value).toFixed(2);
  lrVal.textContent = parseFloat(lrSlider.value).toFixed(2);
  hiddenCountVal.textContent = String(architecture.hiddenSize);
}

function init() {
  initParams();
  initUI();
   updateLayout();
  drawNetwork();
  computeForward();
  computeBackward();
  renderStep();
}

init();
