{-# LANGUAGE FlexibleContexts #-}
module NeuralNetwork where

import Prelude hiding (map, traverse)
import qualified Prelude
import Data.Array.Repa hiding ((++), zipWith)
import qualified Data.Array.Repa as R
import Data.Array.Repa.Algorithms.Matrix (mmultS, row, col)
import System.Random (mkStdGen, randomRs, randomIO)
import Control.Lens hiding (index)

import Debug.Trace (trace)

import Mnist (NormalizedDataSet)
import Util (updateAS, modifyL)
import Functions ( meanSquaredError
                 , crossEntropyError
                 , sigmonoid
                 , softmax
                 , (+^^))

import qualified BackProp as BP

-- $setup
-- >>> import Util ((=~))
-- >>> let y = fromListUnboxed (ix2 2 2) [0.6, 0.9, 0.2, 0.3]
-- >>> let t = fromListUnboxed (ix2 2 2) [0, 1, 1, 0]
-- >>> let x = fromListUnboxed (ix2 2 3) [(1::Double)..6]
-- >>> n <- network [2, 4, 2] (0, 0.1)
-- >>> nb = nnb n

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

network :: [Int] -> (Double, Double) -> IO NN
network (x1:x2:xs) range = do
  n <- network'
  let newLastLayer = last n & _3 .~ softmax
  return $ n & ix (length xs) .~ newLastLayer
  where
    network' =  sequence $ scanl f y xs
    y = g x1 x2
    g :: Int -> Int -> IO Layer
    g a b = do
      r1 <- randomIO
      r2 <- randomIO
      return $ ( matrix (ix2 a b) (a*b) range r1
               , matrix (ix2 1 b) b range r2
               , sigmonoid)
    f :: IO Layer -> Int -> IO Layer
    f a b = do
      l <- a
      g (col.extent.weight $ l) b

nnb :: NN -> NNB
nnb [] = []
nnb [(w, b, _)] = [BP.Affine w b undefined]
nnb ((w, b, _):xs) = [ BP.Affine w b undefined
                     , BP.ReLU undefined] ++ nnb xs

nbn :: NNB -> NN
nbn = concat . fmap f
  where f (BP.Affine w b _) = [(w, b, undefined)]
        f _              = []

predictB :: Matrix D -> NNB ->
            (Matrix D, [BP.Layer BP.Backward])
predictB input n = foldl f (input, []) n
  where f (i, ls) x = (i', s:ls)
          where (BP.OutputMatrix i', s) = BP.forward i x

-- | lossB
-- >>> fst (lossB (y, t) 2 nb) =~ loss (y, t) 2 n $ 1
-- True
lossB :: NormalizedDataSet -> Int -> NNB ->
         (Double, [BP.Layer BP.Backward])
lossB (input, label) _ n = (l, b:bs)
  where
    (x, bs) = predictB (delay input) n
    (BP.OutputDouble l, b) = BP.forward x (BP.SoftmaxWithLoss label undefined)

-- | accuracyB
-- >>> accuracyB (y, t) 2 nb == accuracy (y, t) 2 n
-- True
accuracyB :: NormalizedDataSet -> Int -> NNB -> Double
accuracyB (i, l) bat n = accuracy' l (computeS x) bat
  where
    (x, _) = predictB (delay i) n

-- | gradient
-- >>> gsb = gradient (y, t) 2 nb
-- >>> print $ snd head $ gsb
gradient :: NormalizedDataSet -> Int -> NNB -> Gradients
gradient dataset bat nb= toGradient . backward $ ls
  where
    (_, ls) = lossB dataset bat nb
    backward :: [BP.Layer BP.Backward] -> [BP.Layer BP.Gradient]
    backward = snd . foldl f (undefined, [])
      where f (d, gs) x = (d', g:gs)
              where (d', g) = BP.backward d x
    toGradient :: [BP.Layer BP.Gradient] -> Gradients
    toGradient = concat . fmap f
      where
        f (BP.Affine w b _) = [(toList w, toList b)]
        f _                 = []


predict :: Matrix U -> NN -> Matrix U
predict input n = foldl f input n
  where
    f :: Matrix U -> Layer -> Matrix U
    f i (w, b, a) = a .
                   (+^^b) $ mmultS i w

-- | loss
-- >>> loss (y, t) 2 n =~ 0.6928875360459781 $ 2
-- True
loss :: NormalizedDataSet -> Int -> NN -> Double
loss (input, label) batsize nn = crossEntropyError x label batsize
  where
    x = predict input nn

accuracy :: NormalizedDataSet -> Int -> NN -> Double
accuracy (input, label) batsize nn = accuracy' label
                                               (predict input nn)
                                               batsize

accuracy' :: Matrix U -> Matrix U -> Int -> Double
accuracy' label x batsize = fromIntegral a / fromIntegral batsize
  where
    a = length . filter id $ zipWith (==) (argMax label) (argMax x)
    argMax :: Matrix U -> [Int]
    argMax arr = mapRow (\i -> foldCol (g i) 0 arr)
      where
        g :: Int -> Int -> Int -> Int
        g i y z = let ay = index arr (ix2 i y)
                      az = index arr (ix2 i z)
                  in if ay >= az then y else z
    mapRow f = fmap f [0..r-1]
    foldCol :: (Int -> Int -> Int) -> Int -> Matrix U -> Int
    foldCol f b arr = foldr f b [1..((col.extent$arr)-1)]
    r = row . extent $ label

numericalGradient :: Monad m =>
                     NormalizedDataSet -> Int -> NN -> m Gradients
numericalGradient input bat net = mapM calcGradient [0..len]
  where
    len = length net - 1
    h :: Double
    h = 1e-2
    calcGradient :: Monad m => Int -> m Gradient
    calcGradient i = do
      a <- diffWeights
      b <- diffBiases
      return ( a
             , b)
      where
        f :: Monad m => (Layer, Layer) -> m Double
        f (l, r) = do
          let a = loss input bat (modifyL net i l)
          let b = loss input bat (modifyL net i r)
          return $ (b - a) / (2*h)
        diffWeights, diffBiases :: Monad m => m [Double]
        diffWeights = let l = net !! i
                          w = weight l
                          xs = [0..(size.extent $ w) - 1]
                      in mapM (f . (diffXW l w)) xs
        diffXW :: Layer -> Weight -> Int -> (Layer, Layer)
        diffXW l w n = let newWeight1 = updateAS w n (flip (-) h)
                           newWeight2 = updateAS w n (+h)
                           newLayer1  = l & _1 .~ newWeight1
                           newLayer2  = l & _1 .~ newWeight2
                       in (newLayer1, newLayer2)
        diffBiases = let l = net !! i
                         b = bias l
                         xs = [0..(size.extent $ b) - 1]
                      in mapM (f . (diffXB l b)) xs
        diffXB :: Layer -> Bias -> Int -> (Layer, Layer)
        diffXB l b n = let newBiases1 = updateAS b n (flip (-) h)
                           newBiases2 = updateAS b n (+h)
                           newLayer1  = l & _2 .~ newBiases1
                           newLayer2  = l & _2 .~ newBiases2
                       in (newLayer1, newLayer2)

gradientDescentB :: Monad m => Double -> NNB -> Gradients -> m NNB
gradientDescentB lr nb g = do
  n <- gradientDescent lr (nbn nb) g
  return $ nnb n

gradientDescent :: Monad m => Double -> NN -> Gradients -> m NN
gradientDescent learningRate = (mapM f .) . zip
  where
    f :: Monad m => (Layer, Gradient) -> m Layer
    f ((w, b, a), (gw, gb)) = do
      w' <- computeP $ w -^ gw'
      b' <- computeP $ b -^ gb'
      return (w', b', a)
      where
        shW = extent w
        shB = extent b
        gw', gb' :: Matrix D
        gw' = map (*learningRate) $ fromListUnboxed shW gw
        gb' = map (*learningRate) $ fromListUnboxed shB gb

matrix :: DIM2 -> Int -> (Double, Double) -> Int -> Matrix U
matrix shape len range gen = fromListUnboxed shape $
                             take len . randomRs range $ mkStdGen gen
