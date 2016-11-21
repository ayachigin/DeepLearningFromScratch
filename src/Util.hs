module Util where

import Data.Array.Repa hiding ((++))
import qualified Data.Array.Repa as R
import Data.Vector.Unboxed.Base

updateAS :: (Source r e, Shape sh, Unbox e) =>
            Array r sh e -> Int -> (e -> e) -> Array U sh e
updateAS a i f = modifyAS a i $ f e
  where
    s = extent a
    e = a ! (fromIndex s i)

modifyAS :: (Source r e, Shape sh, Unbox e) =>
            Array r sh e -> Int -> e -> Array U sh e
modifyAS a i v = computeS $ modifyA a i v

modifyA :: (Source r e, Shape sh) =>
          Array r sh e -> Int -> e -> Array D sh e
modifyA arr i v = R.traverse arr id f
  where
    originalShape = extent arr
    f g sh
      | toIndex originalShape sh == i = v
      | otherwise                     = g sh

updateL :: [a] -> Int -> (a -> a) -> [a]
updateL ls i f = modifyL ls i (f (ls !! i))

modifyL :: [a] -> Int -> a -> [a]
modifyL ls i e = case splitAt i ls of
                   (l1, (_:l2)) -> l1++(e:l2)
                   (l1, []) -> init l1 ++ [e]

pickle :: Show a => a -> FilePath -> IO ()
pickle a f = writeFile f . show $ a

unpickle :: Read a => FilePath -> IO a
unpickle = fmap read . readFile

a =~ b = \x -> floor (a*10^x) == floor (b*10^x)
