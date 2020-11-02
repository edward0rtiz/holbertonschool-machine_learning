-- query to get average temperatures per city
SELECT city, AVG(value) AS avg_temp
FROM temperatures GROUP BY city
ORDER BY avg_temp DESC;