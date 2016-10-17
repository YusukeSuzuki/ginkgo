#!/bin/bash

for i in {0..49999} ; do
	FILENAME=`printf "out_%010d.csa" $i`
	if [ -e "sfen/$FILENAME" ] ; then
		echo "cp sfen/$FILENAME train/"
		cp sfen/$FILENAME train/
	fi
done

for i in {50000..70000} ; do
	FILENAME=`printf "out_%010d.csa" $i`
	if [ -e "sfen/$FILENAME" ] ; then
		echo "cp sfen/$FILENAME test/"
		cp sfen/$FILENAME test/
	fi
done

