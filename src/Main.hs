module Main where

type Weight = Double
type Threshold = Double

data Perceptron = Perceptron { w1 :: Double
                             , w2 :: Double
                             , bias :: Double
                             } deriving (Show, Read, Ord, Eq)

main :: IO ()
main = do
  putStrLn "hello world"

andP :: Double -> Double -> Double
andP = perceptron (Perceptron 0.5 0.5 (-0.8))

nandP :: Double -> Double -> Double
nandP = perceptron (Perceptron (-0.5) (-0.5) 0.8)

orP = perceptron (Perceptron 0.1 0.1 0)

notP = perceptron (Perceptron 1 (-1) 0) 1

perceptron :: Perceptron -> Double -> Double -> Double
perceptron (Perceptron w1 w2 b) x y =
  if val <= 0 then
    0
  else
    1
  where
    val = (b+) . sum $ zipWith (*) [x, y] [w1, w2]
