dummy=$0
row=$1
col=$2

if [ $row -lt 0 ] || [ $col -lt 0 ] ; then
       echo "input must be greater than 0 :("
else
       for i in $(seq 1 $row)
       do
             for j in $(seq 1 $col)
             do
                   tmp=`expr $i \* $j`
                   printf "$i * $j = "$tmp
                   printf "      "
             done
             echo ""
       done
fi
exit 0
