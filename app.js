// Small fully-connected neural network visualizer
// Architecture: 2 inputs -> H hidden neurons -> 1 output (all sigmoid).
// H (the number of hidden neurons) can be changed with a slider.

const svg = document.getElementById("network-svg");
const stepTitleEl = document.getElementById("step-title");
const equationsEl = document.getElementById("equations");
const stepExplainerEl = document.getElementById("step-explainer");
const valuesTableEl = document.getElementById("values-table");
const variableLegendEl = document.getElementById("variable-legend");

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
// We keep a small configuration object so the same code works
// for different hidden-layer sizes.
let architecture = {
  inputCount: 2,
  hiddenNeuronCount: 2,
  outputCount: 1,
};

// Weights and biases are stored in arrays for clarity:
// - inputToHiddenWeights[hiddenIndex][inputIndex]  (w_ih[j,i])
// - hiddenBiases[hiddenIndex]                     (b_h[j])
// - hiddenToOutputWeights[hiddenIndex]            (w_ho[j])
// - outputBias                                    (b_o)
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

function renderVariableLegend() {
  if (!variableLegendEl) return;
  variableLegendEl.innerHTML = `
    <div class="variable-legend-item"><strong>x₁, x₂</strong>: inputs you choose with the sliders.</div>
    <div class="variable-legend-item"><strong>y</strong>: target output you want the network to match.</div>
    <div class="variable-legend-item"><strong>ŷ</strong>: network's prediction after the forward pass.</div>
    <div class="variable-legend-item"><strong>hⱼ</strong>: activation of hidden neuron j (after sigmoid).</div>
    <div class="variable-legend-item"><strong>w_ih[j,i]</strong>: weight from input i into hidden neuron j.</div>
    <div class="variable-legend-item"><strong>b_h[j]</strong>: bias added to hidden neuron j before sigmoid.</div>
    <div class="variable-legend-item"><strong>w_ho[j]</strong>: weight from hidden neuron j into the output.</div>
    <div class="variable-legend-item"><strong>b_o</strong>: bias added to the output neuron before sigmoid.</div>
    <div class="variable-legend-item"><strong>L</strong>: loss (error) measuring how far ŷ is from y.</div>
    <div class="variable-legend-item"><strong>dL/d·</strong>: gradient of the loss with respect to some value.</div>
  `;
}

function randomWeight() {
  return (Math.random() * 2 - 1) * 0.8; // in [-0.8, 0.8]
}

function initParams() {
  const hiddenNeuronCount = architecture.hiddenNeuronCount;
  const inputCount = architecture.inputCount;

  const inputToHiddenWeights = Array.from({ length: hiddenNeuronCount }, () =>
    Array.from({ length: inputCount }, () => randomWeight())
  );
  const hiddenBiases = Array.from({ length: hiddenNeuronCount }, () => randomWeight());

  const hiddenToOutputWeights = Array.from({ length: hiddenNeuronCount }, () => randomWeight());
  const outputBias = randomWeight();

  params = { inputToHiddenWeights, hiddenBiases, hiddenToOutputWeights, outputBias };
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidPrime(x) {
  const s = sigmoid(x);
  return s * (1 - s);
}

function fmt(x, digits = 4) {
  return Number(x).toFixed(digits);
}

function computeForward() {
  const inputVector = [parseFloat(inputX1.value), parseFloat(inputX2.value)];
  const targetOutput = parseFloat(targetY.value);

  const hiddenNeuronCount = architecture.hiddenNeuronCount;
  const hiddenWeightedSums = new Array(hiddenNeuronCount); // z_h
  const hiddenActivations = new Array(hiddenNeuronCount);  // h

  for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronCount; hiddenIndex++) {
    let sum = 0;
    for (let inputIndex = 0; inputIndex < architecture.inputCount; inputIndex++) {
      sum += params.inputToHiddenWeights[hiddenIndex][inputIndex] * inputVector[inputIndex];
    }
    sum += params.hiddenBiases[hiddenIndex];
    hiddenWeightedSums[hiddenIndex] = sum;
    hiddenActivations[hiddenIndex] = sigmoid(sum);
  }

  let outputWeightedSum = 0; // z_o
  for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronCount; hiddenIndex++) {
    outputWeightedSum += params.hiddenToOutputWeights[hiddenIndex] * hiddenActivations[hiddenIndex];
  }
  outputWeightedSum += params.outputBias;
  const predictedOutput = sigmoid(outputWeightedSum);

  const loss = 0.5 * Math.pow(predictedOutput - targetOutput, 2); // squared error

  cache = {
    inputVector,
    targetOutput,
    hiddenWeightedSums,
    hiddenActivations,
    outputWeightedSum,
    predictedOutput,
    loss,
  };
}

function computeBackward() {
  const {
    inputVector,
    targetOutput,
    hiddenWeightedSums,
    hiddenActivations,
    outputWeightedSum,
    predictedOutput,
  } = cache;

  const hiddenNeuronCount = architecture.hiddenNeuronCount;

  const gradLossWrtPrediction = predictedOutput - targetOutput; // dL/dŷ
  const gradPredictionWrtOutputSum = sigmoidPrime(outputWeightedSum); // dŷ/dz_out
  const gradLossWrtOutputSum = gradLossWrtPrediction * gradPredictionWrtOutputSum; // dL/dz_out

  const gradLossWrtHiddenToOutputWeights = new Array(hiddenNeuronCount); // dL/dw_ho[j]
  const gradLossWrtHiddenActivations = new Array(hiddenNeuronCount);     // dL/dh[j]
  const gradHiddenActivationWrtHiddenSum = new Array(hiddenNeuronCount); // σ'(z_h[j])
  const gradLossWrtHiddenWeightedSums = new Array(hiddenNeuronCount);    // dL/dz_h[j]

  const gradLossWrtInputToHiddenWeights = Array.from(
    { length: hiddenNeuronCount },
    () => new Array(architecture.inputCount)
  );
  const gradLossWrtHiddenBiases = new Array(hiddenNeuronCount);

  for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronCount; hiddenIndex++) {
    gradLossWrtHiddenToOutputWeights[hiddenIndex] =
      gradLossWrtOutputSum * hiddenActivations[hiddenIndex];

    gradLossWrtHiddenActivations[hiddenIndex] =
      gradLossWrtOutputSum * params.hiddenToOutputWeights[hiddenIndex];

    gradHiddenActivationWrtHiddenSum[hiddenIndex] = sigmoidPrime(
      hiddenWeightedSums[hiddenIndex]
    );

    gradLossWrtHiddenWeightedSums[hiddenIndex] =
      gradLossWrtHiddenActivations[hiddenIndex] *
      gradHiddenActivationWrtHiddenSum[hiddenIndex];

    for (let inputIndex = 0; inputIndex < architecture.inputCount; inputIndex++) {
      gradLossWrtInputToHiddenWeights[hiddenIndex][inputIndex] =
        gradLossWrtHiddenWeightedSums[hiddenIndex] * inputVector[inputIndex];
    }

    gradLossWrtHiddenBiases[hiddenIndex] = gradLossWrtHiddenWeightedSums[hiddenIndex];
  }

  const gradLossWrtOutputBias = gradLossWrtOutputSum;

  Object.assign(cache, {
    gradLossWrtPrediction,
    gradPredictionWrtOutputSum,
    gradLossWrtOutputSum,
    gradLossWrtHiddenToOutputWeights,
    gradLossWrtOutputBias,
    gradLossWrtHiddenActivations,
    gradHiddenActivationWrtHiddenSum,
    gradLossWrtHiddenWeightedSums,
    gradLossWrtInputToHiddenWeights,
    gradLossWrtHiddenBiases,
  });
}

function applyGradientDescent() {
  const lr = parseFloat(lrSlider.value);
  const hiddenNeuronCount = architecture.hiddenNeuronCount;

  const {
    gradLossWrtInputToHiddenWeights,
    gradLossWrtHiddenBiases,
    gradLossWrtHiddenToOutputWeights,
    gradLossWrtOutputBias,
  } = cache;

  for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronCount; hiddenIndex++) {
    for (let inputIndex = 0; inputIndex < architecture.inputCount; inputIndex++) {
      params.inputToHiddenWeights[hiddenIndex][inputIndex] -=
        lr * gradLossWrtInputToHiddenWeights[hiddenIndex][inputIndex];
    }
    params.hiddenBiases[hiddenIndex] -= lr * gradLossWrtHiddenBiases[hiddenIndex];
    params.hiddenToOutputWeights[hiddenIndex] -=
      lr * gradLossWrtHiddenToOutputWeights[hiddenIndex];
  }
  params.outputBias -= lr * gradLossWrtOutputBias;
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

  const hiddenNeuronCount = architecture.hiddenNeuronCount;
  const xHidden = 280;
  const top = 60;
  const bottom = 240;
  const hidden = [];

  if (hiddenNeuronCount === 1) {
    hidden.push({ id: "h1", x: xHidden, y: 150 });
  } else {
    const gap = (bottom - top) / (hiddenNeuronCount - 1);
    for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronCount; hiddenIndex++) {
      hidden.push({ id: `h${hiddenIndex + 1}`, x: xHidden, y: top + gap * hiddenIndex });
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
        .conn-bias { stroke: #38bdf8; stroke-width: 1.6; stroke-dasharray: 4 3; }
        .neuron { fill: #020617; stroke-width: 2; }
        .neuron-input { stroke: #22c55e; }
        .neuron-hidden { stroke: #eab308; }
        .neuron-output { stroke: #f97316; }
        .neuron-bias { stroke: #38bdf8; }
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

  // Bias node for hidden layer (constant 1 going into every hidden neuron)
  const hiddenBiasX = 210;
  const hiddenBiasY = 40;
  const hiddenBiasCircle = createCircle(hiddenBiasX, hiddenBiasY, 10, "neuron neuron-bias");
  svg.appendChild(hiddenBiasCircle);
  svg.appendChild(createText(hiddenBiasX, hiddenBiasY, "1", "neuron-text"));

  layout.hidden.forEach((hNode) => {
    const line = createLine(hiddenBiasX + 14, hiddenBiasY, hNode.x - 22, hNode.y, "conn-bias");
    svg.appendChild(line);
  });

  // Connections hidden -> output
  layout.hidden.forEach((hNode) => {
    const yNode = layout.output[0];
    const line = createLine(hNode.x + 22, hNode.y, yNode.x - 22, yNode.y, "conn");
    line.dataset.from = hNode.id;
    line.dataset.to = yNode.id;
    svg.appendChild(line);
  });

  // Bias node for output neuron (constant 1 going into output)
  const outputBiasX = 410;
  const outputBiasY = 40;
  const outputBiasCircle = createCircle(outputBiasX, outputBiasY, 10, "neuron neuron-bias");
  svg.appendChild(outputBiasCircle);
  svg.appendChild(createText(outputBiasX, outputBiasY, "1", "neuron-text"));

  const yNode = layout.output[0];
  const outputBiasLine = createLine(outputBiasX + 14, outputBiasY, yNode.x - 22, yNode.y, "conn-bias");
  svg.appendChild(outputBiasLine);

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

// UI rendering for equations & values
function renderStep() {
  computeForward();
  computeBackward();
  highlightForStep(currentStep);

  stepItems.forEach((item, idx) => {
    item.classList.toggle("active", idx === currentStep);
  });

  const s = cache;
  const hiddenNeuronCount = architecture.hiddenNeuronCount;

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
      "again to get the final prediction ŷ.";

    const pairs = [
      ["Input x₁", s.inputVector[0]],
      ["Input x₂", s.inputVector[1]],
    ];
    for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronCount; hiddenIndex++) {
      pairs.push([
        `Hidden ${hiddenIndex + 1} sum (z_h${hiddenIndex + 1})`,
        s.hiddenWeightedSums[hiddenIndex],
      ]);
      pairs.push([
        `Hidden ${hiddenIndex + 1} act (h${hiddenIndex + 1})`,
        s.hiddenActivations[hiddenIndex],
      ]);
    }
    pairs.push(["Output sum (z_out)", s.outputWeightedSum]);
    pairs.push(["Prediction ŷ", s.predictedOutput]);
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
      ["Prediction ŷ", s.predictedOutput],
      ["Target y", s.targetOutput],
      ["Loss L", s.loss],
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
      ["Gradient dL/dŷ (loss vs prediction)", s.gradLossWrtPrediction],
      ["σ'(z_out) (slope of sigmoid)", s.gradPredictionWrtOutputSum],
      ["Gradient dL/dz_out", s.gradLossWrtOutputSum],
    ];
    for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronCount; hiddenIndex++) {
      pairs.push([
        `Grad dL/dw_ho${hiddenIndex + 1} (hidden ${hiddenIndex + 1} → output)`,
        s.gradLossWrtHiddenToOutputWeights[hiddenIndex],
      ]);
    }
    pairs.push(["Grad dL/db_o (output bias)", s.gradLossWrtOutputBias]);
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
    for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronCount; hiddenIndex++) {
      pairs.push([
        `Grad dL/dh${hiddenIndex + 1} (hidden act)`,
        s.gradLossWrtHiddenActivations[hiddenIndex],
      ]);
      pairs.push([
        `σ'(z_h${hiddenIndex + 1}) (hidden slope)`,
        s.gradHiddenActivationWrtHiddenSum[hiddenIndex],
      ]);
      pairs.push([
        `Grad dL/dz_h${hiddenIndex + 1} (hidden sum)`,
        s.gradLossWrtHiddenWeightedSums[hiddenIndex],
      ]);
    }
    renderValues(pairs);
  } else if (currentStep === 4) {
    // Snapshot old weights and loss, then take one update step
    const oldLoss = s.loss;
    const oldParams = JSON.parse(JSON.stringify(params));

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
      "that reduces the loss. Repeating this step many times is what training is.";

    stepExplainerEl.textContent =
      "Below you can see every weight and bias before and after " +
      "this step (shown as old → new). Each step nudges them a bit.";

    const v = [];
    v.push(["Old loss", oldLoss]);
    v.push(["New loss", newLoss]);

    // Show how each weight and bias changed: old → new
    for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronCount; hiddenIndex++) {
      for (let inputIndex = 0; inputIndex < architecture.inputCount; inputIndex++) {
        v.push([
          `w_ih${hiddenIndex + 1},${inputIndex + 1}`,
          `${fmt(oldParams.inputToHiddenWeights[hiddenIndex][inputIndex])} → ` +
            `${fmt(params.inputToHiddenWeights[hiddenIndex][inputIndex])}`,
        ]);
      }
      v.push([
        `b_h${hiddenIndex + 1}`,
        `${fmt(oldParams.hiddenBiases[hiddenIndex])} → ` +
          `${fmt(params.hiddenBiases[hiddenIndex])}`,
      ]);
    }

    for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronCount; hiddenIndex++) {
      v.push([
        `w_ho${hiddenIndex + 1}`,
        `${fmt(oldParams.hiddenToOutputWeights[hiddenIndex])} → ` +
          `${fmt(params.hiddenToOutputWeights[hiddenIndex])}`,
      ]);
    }
    v.push([
      "b_o",
      `${fmt(oldParams.outputBias)} → ${fmt(params.outputBias)}`,
    ]);

    renderValues(v, ["Old loss", "New loss"]);
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
    if (typeof value === "number") {
      valEl.textContent = fmt(value);
    } else {
      valEl.textContent = String(value);
    }

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
  architecture.hiddenNeuronCount = H;
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
  hiddenCountVal.textContent = String(architecture.hiddenNeuronCount);
}

function init() {
  initParams();
  initUI();
  updateLayout();
  drawNetwork();
  renderVariableLegend();
  computeForward();
  computeBackward();
  renderStep();
}

init();
