module Types where

import Data.Array.Repa
import Control.Lens hiding (index)

import qualified BackProp as BP

type Matrix r = Array r DIM2 Double

type Vector = Array U DIM1 Double

type Weight = Matrix U

type Bias = Matrix U

type ActivationFunction = Matrix D -> Matrix U

type Layer = (Weight, Bias, ActivationFunction)

type LayerB = BP.Layer

type Gradient = ([Double], [Double])

type Gradients = [Gradient]

weight :: Layer -> Weight
weight = (^._1)

bias :: Layer -> Bias
bias = (^._2)

type NN = [Layer]

type NNB = [BP.Layer BP.Forward]

type LossFunction = Matrix U -> Matrix U -> Int -> Double
