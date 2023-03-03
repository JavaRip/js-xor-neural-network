const RVAL = 1000000 // round numbers to
const e = 2.71828 // Euler's number
const η = 0.1 // learning rate
let α = 0 // momentum

let Δw1Old = [0, 0, 0, 0, 0, 0]
let Δw2Old = [0, 0, 0]

const INPUT = [
  [1, 1, 1],
  [1, 1, 0],
  [1, 0, 1],
  [1, 0, 0],
]

const W1 = [
  [-0.9, 0.35],
  [-0.2, -0.1],
  [0.9, -0.25],
]

const W2 = [
  [0.4],
  [-0.3],
  [-0.9],
]

const INNODES = [1, NaN, NaN]
const HIDNODES = [1, NaN, NaN]
const OUTNODES = [NaN, NaN, NaN]
const TARGET = [0, 1, 1, 0]

function activation(xbar) {
  return 1 / (1 + Math.pow(e, -xbar)) 
}

function dotProduct(inputs, weights) {
  const ret = []
  for (let j = 0; j < weights[0].length; j += 1) {
    let sum = 0
    for (let i = 0; i < inputs.length; i += 1) {
      sum += weights[i][j] * inputs[i]
    }
    ret.push(sum)
  }

  return ret
}

function activateBar(bar) {
  const ret = []
  for (const val of bar) {
    ret.push(activation(val))
  }
  return ret
}

function yError(y, t) {
  return y[0] * (1 - y[0]) * (t - y[0])
}

function hError(h, ye, w) {
  const ret = []
  for (let j = 1; j < h.length; j += 1) {
    ret.push(h[j] * (1 - h[j]) * ye * w[j])
  }
  return ret
}

function deltaW2(ye, h) {
  const ret = []
  for (let j = 0; j < h.length; j += 1) {
    ret.push(η * ye * h[j] + (α * Δw2Old[0]))
  }
  return ret
}

function deltaW1(he, inp) {
  const ret = []
  // for each input value
  let deltaIndex = 0
  for (let j = 0; j < inp.length; j += 1) {
    // for each hidden node
    for (let h = 0; h < he.length; h += 1) {
      // learning rate * hidden error[index] * input value [index] + momentum * old weights [index][index]
      ret.push(η * he[h] * inp[j] + (α * Δw1Old[deltaIndex]))
      deltaIndex += 1
    }
  }
  return ret
}

function epoch() {
  const outputs = []
  const yeArr = []
  const heArr = []
  const hArr = []
  const kbarArr = []
  const hbarArr = []
  const deltaW2Arr = []
  const deltaW1Arr = []

  for (let i = 0; i < INPUT.length; i += 1) {
    const input = INPUT[i]
    const target = TARGET[i]

    // get weighted sum of values into hidden nodes
    const kbar = dotProduct(input, W1)
    kbarArr.push(kbar)

    // put weighted sum of values through activation function
    const h = activateBar(kbar)

    // add bias to h
    h.unshift(1)
    hArr.push(h)

    // get weighted sum of values into output node
    const hbar = dotProduct(h, W2)
    hbarArr.push(hbar)

    // put weighted sum of values through activation function
    const y = activateBar(hbar)
    outputs.push(y)

    // calculate error at output
    const ye = yError(y, target)
    yeArr.push(ye)

    // calculate error at hidden layer
    const he = hError(h, ye, W2)
    heArr.push(he)

    // calculate the change in weights from hidden to output
    const Δw2 = deltaW2(ye, h)
    deltaW2Arr.push(Δw2)
    Δw2Old = Δw2

    // calculate the change in weights from input to hidden
    const Δw1 = deltaW1(he, input)
    deltaW1Arr.push(Δw1)
    Δw1Old = Δw1

    // update the weights
    let deltaIndex = 0
    for (let i = 0; i < W1.length; i += 1) {
      for (let j = 0; j < W1[0].length; j+= 1) {
        W1[i][j] = W1[i][j] + Δw1[deltaIndex]
        deltaIndex += 1  
      }
    }

    for (let i = 0; i < W2.length; i += 1) {
      W2[i][0] = W2[i][0] + Δw2[i]
    }
  }

  console.log('input nodes')
  console.log(INPUT)
  console.log('W1')
  console.log(W1)
  console.log('xbar')
  console.log(kbarArr)
  console.log('h')
  console.log(hArr)
  console.log('W2')
  console.log(W2)
  console.log('hbar')
  console.log(hbarArr)
  console.log('y')
  console.log(outputs)
  console.log('target')
  console.log(TARGET)
  console.log('ye / e2')
  console.log(yeArr)
  console.log('he / e1')
  console.log(heArr)
  console.log('delta W2')
  console.log(deltaW2Arr)
  console.log('delta W1')
  console.log(deltaW1Arr)
}

const EPOCHS = 500
for (let i = 0; i < EPOCHS; i += 1) {
  epoch()
  console.log('================================')
}