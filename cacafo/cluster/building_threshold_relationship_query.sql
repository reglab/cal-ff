WITH "buildingthresholdrelationship" AS (
	SELECT DISTINCT -- select all buildings matching distance, fuzzy, and tfidf
		"distancerelationship"."building_id" AS "building_id",
		"distancerelationship"."other_building_id" AS "other_building_id"
		FROM "buildingrelationship" as "distancerelationship"
		INNER JOIN (
			SELECT
				"buildingrelationship"."building_id",
				"buildingrelationship"."other_building_id"
			FROM "buildingrelationship"
			WHERE (
				("buildingrelationship"."reason" = 'parcel name tf-idf')
				AND ("buildingrelationship"."weight" > {tfidf})
				AND ("buildingrelationship"."weight" < {tfidf_max})
			)
		) AS "parcelnamerelationship" ON (
			("parcelnamerelationship"."building_id" = "distancerelationship"."building_id")
			AND ("parcelnamerelationship"."other_building_id" = "distancerelationship"."other_building_id")
		)
		INNER JOIN (
			SELECT
				"buildingrelationship"."building_id",
				"buildingrelationship"."other_building_id"
			FROM "buildingrelationship"
			WHERE (
				("buildingrelationship"."reason" = 'parcel name fuzzy')
				AND ("buildingrelationship"."weight" > {fuzzy})
				AND ("buildingrelationship"."weight" < {fuzzy_max})
			)
		) AS "parcelfuzzyrelationship" ON (
			("parcelfuzzyrelationship"."building_id" = "distancerelationship"."building_id")
			AND ("parcelfuzzyrelationship"."other_building_id" = "distancerelationship"."other_building_id")
		)
		WHERE (
			("distancerelationship"."reason" = 'distance')
			AND ("distancerelationship"."weight" > (1000 - {distance}))
		)
	UNION SELECT DISTINCT -- select all buildings with manual parcel name annotation overrides
		"t1"."building_id" AS "building_id",
		"t1"."other_building_id" AS "other_building_id"
		FROM "buildingrelationship" AS "t1"
		INNER JOIN (
			SELECT
				"buildingrelationship"."building_id",
				"buildingrelationship"."other_building_id"
			FROM "buildingrelationship"
			WHERE (
				("buildingrelationship"."reason" = 'distance')
				AND ("buildingrelationship"."weight" > (1000 - {distance}))
			)
		) AS "distancerelationship" ON (
			("distancerelationship"."building_id" = "t1"."building_id")
			AND ("distancerelationship"."other_building_id" = "t1"."other_building_id")
		)
		WHERE ("t1"."reason" = 'parcel name annotation')
	UNION SELECT DISTINCT -- select all buildings within distance radius and no parcel name
		"br"."building_id" AS "building_id",
		"br"."other_building_id" AS "other_building_id"
		FROM "buildingrelationship" AS "br"
		JOIN "building" AS "b" ON ("b"."id" = "br"."building_id")
		LEFT JOIN "parcel" AS "p" ON ("p"."id" = "b"."parcel_id")
		WHERE (
			("br"."reason" = 'distance')
			AND ("br"."weight" > (1000 - {no_owner_distance}))
			AND (("p"."owner" IS NULL) OR ("p"."owner" = ''))
		)
	UNION SELECT DISTINCT -- select all buildings with a matching parcel
		"t3"."building_id" AS "building_id",
		"t3"."other_building_id" AS "other_building_id"
		FROM "buildingrelationship" AS "t3" WHERE ("t3"."reason" = 'matching parcel')
), "lonebuildingrelationships" AS (
	SELECT DISTINCT
		"buildingrelationship"."building_id" AS "building_id",
		"buildingrelationship"."other_building_id" AS "other_building_id"
		FROM "buildingrelationship"
		LEFT JOIN "buildingthresholdrelationship" ON (
			("buildingthresholdrelationship"."building_id" = "buildingrelationship"."building_id")
		)
		WHERE (
			("buildingthresholdrelationship"."building_id" IS NULL)
			AND ("buildingrelationship"."reason" = 'distance')
			AND ("buildingrelationship"."weight" > (1000 - {lone_building_distance}))
		)
)
SELECT DISTINCT
	"building_id",
	"other_building_id"
	FROM "buildingthresholdrelationship"
UNION SELECT DISTINCT
	"building_id",
	"other_building_id"
	FROM "lonebuildingrelationships"
