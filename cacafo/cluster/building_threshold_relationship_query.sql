WITH "buildingthresholdrelationship" AS (
	SELECT DISTINCT -- select all buildings matching distance, fuzzy, and tfidf
		"distancerelationship"."building_id" AS "building_id",
		"distancerelationship"."related_building_id" AS "related_building_id"
		FROM "building_relationship" as "distancerelationship"
		INNER JOIN (
			SELECT
				"building_relationship"."building_id",
				"building_relationship"."related_building_id"
			FROM "building_relationship"
			WHERE (
				("building_relationship"."reason" = 'parcel name tf-idf')
				AND ("building_relationship"."weight" > {tfidf})
				AND ("building_relationship"."weight" < {tfidf_max})
			)
		) AS "parcelnamerelationship" ON (
			("parcelnamerelationship"."building_id" = "distancerelationship"."building_id")
			AND ("parcelnamerelationship"."related_building_id" = "distancerelationship"."related_building_id")
		)
		INNER JOIN (
			SELECT
				"building_relationship"."building_id",
				"building_relationship"."related_building_id"
			FROM "building_relationship"
			WHERE (
				("building_relationship"."reason" = 'parcel name fuzzy')
				AND ("building_relationship"."weight" > {fuzzy})
				AND ("building_relationship"."weight" < {fuzzy_max})
			)
		) AS "parcelfuzzyrelationship" ON (
			("parcelfuzzyrelationship"."building_id" = "distancerelationship"."building_id")
			AND ("parcelfuzzyrelationship"."related_building_id" = "distancerelationship"."related_building_id")
		)
		WHERE (
			("distancerelationship"."reason" = 'distance')
			AND ("distancerelationship"."weight" > (1000 - {distance}))
		)
	UNION SELECT DISTINCT -- select all buildings with manual parcel name annotation overrides
		"t1"."building_id" AS "building_id",
		"t1"."related_building_id" AS "related_building_id"
		FROM "building_relationship" AS "t1"
		INNER JOIN (
			SELECT
				"building_relationship"."building_id",
				"building_relationship"."related_building_id"
			FROM "building_relationship"
			WHERE (
				("building_relationship"."reason" = 'distance')
				AND ("building_relationship"."weight" > (1000 - {distance}))
			)
		) AS "distancerelationship" ON (
			("distancerelationship"."building_id" = "t1"."building_id")
			AND ("distancerelationship"."related_building_id" = "t1"."related_building_id")
		)
		WHERE (
			("t1"."reason" = 'parcel owner annotation')
			AND ("t1"."weight" = 1000)
		)
	UNION SELECT DISTINCT -- select all buildings within distance radius and no parcel name
		"br"."building_id" AS "building_id",
		"br"."related_building_id" AS "related_building_id"
		FROM "building_relationship" AS "br"
		JOIN "building" AS "b" ON ("b"."id" = "br"."building_id")
		LEFT JOIN "parcel" AS "p" ON ("p"."id" = "b"."parcel_id")
		WHERE (
			("br"."reason" = 'distance')
			AND ("br"."weight" > (1000 - {no_owner_distance}))
			AND (("p"."owner" IS NULL) OR ("p"."owner" = ''))
		)
	UNION SELECT DISTINCT -- select all buildings with a matching parcel
		"t3"."building_id" AS "building_id",
		"t3"."related_building_id" AS "related_building_id"
		FROM "building_relationship" AS "t3" WHERE ("t3"."reason" = 'matching parcel')
), "lonebuilding_relationships" AS (
	SELECT DISTINCT
		"building_relationship"."building_id" AS "building_id",
		"building_relationship"."related_building_id" AS "related_building_id"
		FROM "building_relationship"
		LEFT JOIN "buildingthresholdrelationship" ON (
			("buildingthresholdrelationship"."building_id" = "building_relationship"."building_id")
		)
		WHERE (
			("buildingthresholdrelationship"."building_id" IS NULL)
			AND ("building_relationship"."reason" = 'distance')
			AND ("building_relationship"."weight" > (1000 - {lone_building_distance}))
		)
)
SELECT DISTINCT
	"building_id",
	"related_building_id"
	FROM "buildingthresholdrelationship"
UNION SELECT DISTINCT
	"building_id",
	"related_building_id"
	FROM "lonebuilding_relationships"
