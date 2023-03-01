const RVAL = 1000000 // round numbers to
const e = 2.71828 // Euler's number
const η = 0.1 // learning rate
let α = 0 // momentum
let Δw2Old = 0
let Δw1Old = 0

const INPUT = [
  [1, 1, 1],
  [1, 1, 0],
  [1, 0, 1],
  [1, 0, 0],
]

const W1 = [
  [0.5, 0.35],
  [0.2, -0.1],
  [0.15, -0.25],
]

const W2 = [
  [0.4],
  [0.3],
  [0.6],
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
    ret.push(η * ye * h[j] + α * Δw2Old)
  }
  return ret
}

function deltaW1(he, inp) {
  const ret = []
  // for each input value
  for (let j = 0; j < inp.length; j += 1) {
    // for each hidden node
    for (let h = 0; h < he.length; h += 1) {
      // learning rate * hidden error[index] * input value [index] + momentum * old weights [index][index]
      ret.push(η * he[h] * inp[j] + α * Δw1Old/*[j][h]*/)
    }
  }
  return ret
}

function epoch() {
  const outputs = []
  const yeArr = []
  for (let i = 0; i < INPUT.length; i += 1) {
    const input = INPUT[i]
    // get weighted sum of values into hidden nodes
    const kbar = dotProduct(input, W1)

    // put weighted sum of values through activation function
    const h = activateBar(kbar)

    // add bias to h
    h.unshift(1)

    // get weighted sum of values into output node
    const hbar = dotProduct(h, W2)

    // put weighted sum of values through activation function
    const y = activateBar(hbar)
    outputs.push(y)

    // calculate error at output
    const ye = yError(y, TARGET[i])
    yeArr.push(ye)

    // calculate error at hidden layer
    const he = hError(h, ye, W2)

    // calculate the change in weights from hidden to output
    const Δw2 = deltaW2(ye, h)

    // calculate the change in weights from input to hidden
    const Δw1 = deltaW1(he, input)

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

  console.log('input:')
  console.log(INPUT)
  console.log('\noutput:')
  console.log(outputs)
  console.log('\ntarget:')
  console.log(TARGET)
  console.log('\nerror:')
  console.log(yeArr)
  console.log('\nAverage err:')
  console.log(yeArr.reduce((acc, x) => acc + x) / yeArr.length)
}

const EPOCHS = 1000
console.log(W1)
console.log(W2)
for (let i = 0; i < EPOCHS; i += 1) {
  epoch()
}
console.log(W1)
console.log(W2)