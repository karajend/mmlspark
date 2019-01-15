// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.util.UUID

import com.microsoft.ml.spark.RESTHelpers._
import org.apache.http.client.methods.HttpDelete
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{typedLit, lit}

import scala.collection.mutable


trait HasAzureSearchKey {
  lazy val azureSearchKey = sys.env("AZURE_SEARCH_KEY")
}

class SearchWriterSuite extends TestBase with HasAzureSearchKey with IndexLister
  with TransformerFuzzing[AddDocuments]{

  import session.implicits._

  private val testServiceName = "mmlspark-azure-search"  //" metartworksindex"

  private def createTestData(numDocs: Int): DataFrame = {
    (0 until numDocs)
      .map(i => ("upload", s"$i", s"file$i", s"text$i"))
      .toDF("searchAction", "id", "fileName", "text")
  }

  private def createSimpleIndexJson(indexName: String): String = {
    s"""
       |{
       |    "name": "$indexName",
       |    "fields": [
       |      {
       |        "name": "id",
       |        "type": "Edm.String",
       |        "key": true,
       |        "facetable": false
       |      },
       |    {
       |      "name": "fileName",
       |      "type": "Edm.String",
       |      "searchable": false,
       |      "sortable": false,
       |      "facetable": false
       |    },
       |    {
       |      "name": "text",
       |      "type": "Edm.String",
       |      "filterable": false,
       |      "sortable": false,
       |      "facetable": false
       |    }
       |    ]
       |  }
    """.stripMargin
  }

  private def createMetIndexTest(indexName: String): String = {
    s"""
       |{
       |"name": "$indexName",
       |    "fields": [
       |        {
       |            "name": "ObjectID",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false,
       |            "key": true
       |        },
       |        {
       |            "name": "Department",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "Title",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "Culture",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "Medium",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "Classification",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "LinkResource",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "PrimaryImageUrl",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "Neighbors",
       |            "type": "Collection(Edm.String)",
       |            "searchable": false,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": false,
       |            "facetable": false
       |        }
       |    ]
       |}
     """.stripMargin
  }

  private def createMetIndex(indexName: String): String = {
    s"""
       |{
       |	"name": "$indexName",
       |    "fields": [
       |        {
       |            "name": "ObjectNumber",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "IsHighlight",
       |            "type": "Edm.Boolean",
       |            "searchable": false,
       |            "filterable": false,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "IsPublicDomain",
       |            "type": "Edm.Boolean",
       |            "searchable": false,
       |            "filterable": false,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "ObjectID",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false,
       |            "key": true
       |        },
       |        {
       |            "name": "Department",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "ObjectName",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Title",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Culture",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "Period",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Dynasty",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Reign",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Portfolio",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "ArtistRole",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "ArtistPrefix",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "ArtistDisplayName",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "ArtistDisplayBio",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "ArtistSuffix",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "ArtistAlphaSort",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "ArtistNationality",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "ArtistBeginDate",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "ArtistEndDate",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "ObjectDate",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "ObjectBeginDate",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "ObjectEndDate",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Medium",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "Dimensions",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "CreditLine",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "GeographyType",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "City",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "State",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "County",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Country",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Region",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Subregion",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Locale",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Locus",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Excavation",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "River",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Classification",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": true
       |        },
       |        {
       |            "name": "RightsandReproduction",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "LinkResource",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "MetadataDate",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Repository",
       |            "type": "Edm.String",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Tags",
       |            "type": "Collection(Edm.String)",
       |            "searchable": true,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": false,
       |            "facetable": false
       |        },
       |        {
       |            "name": "PrimaryImageUrl",
       |            "type": "Edm.String",
       |            "searchable": false,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": true,
       |            "facetable": false
       |        },
       |        {
       |            "name": "AdditionalImageUrls",
       |            "type": "Collection(Edm.String)",
       |            "searchable": false,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": false,
       |            "facetable": false
       |        },
       |        {
       |            "name": "Neighbors",
       |            "type": "Collection(Edm.String)",
       |            "searchable": false,
       |            "filterable": true,
       |            "retrievable": true,
       |            "sortable": false,
       |            "facetable": false
       |        }
       |    ]
       |}
     """.stripMargin
  }

  private val createdIndexes: mutable.ListBuffer[String] = mutable.ListBuffer()

  private def generateIndexName(): String = {
    val name = s"test-${UUID.randomUUID().hashCode()}"
    createdIndexes.append(name)
    name
  }

//  override def afterAll(): Unit = {
//    //TODO make this existing search indices when multiple builds are allowed
//    println("Cleaning up services")
//    val successfulCleanup = getExisting(azureSearchKey, testServiceName).map { n =>
//      val deleteRequest = new HttpDelete(
//        s"https://$testServiceName.search.windows.net/indexes/$n?api-version=2017-11-11")
//      deleteRequest.setHeader("api-key", azureSearchKey)
//      val response = safeSend(deleteRequest)
//      response.getStatusLine.getStatusCode
//    }.forall(_ == 204)
//    super.afterAll()
//    assert(successfulCleanup)
//    ()
//  }

  private def retryWithBackoff[T](f: => T,
                                  timeouts: List[Long] =
                                  List(5000, 10000, 50000, 100000, 200000, 200000)): T = {
    try {
      f
    } catch {
      case _: Exception if timeouts.nonEmpty =>
        println(s"Sleeping for ${timeouts.head}")
        Thread.sleep(timeouts.head)
        retryWithBackoff(f, timeouts.tail)
    }
  }

  lazy val df4: DataFrame = createTestData(4)
  lazy val df10: DataFrame = createTestData(10)
  lazy val bigDF: DataFrame = createTestData(10000)

  override val sortInDataframeEquality: Boolean = true

  lazy val ad: AddDocuments = {
    val in = generateIndexName()
    SearchIndex.createIfNoneExists(azureSearchKey,
      testServiceName,
      createSimpleIndexJson(in))
    new AddDocuments()
      .setSubscriptionKey(azureSearchKey)
      .setServiceName(testServiceName)
      .setOutputCol("out").setErrorCol("err")
      .setIndexName(in)
      .setActionCol("searchAction")
  }

  override def testObjects(): Seq[TestObject[AddDocuments]] =
    Seq(new TestObject(ad, df4))

  override def reader: MLReadable[_] = AddDocuments

  def writeHelper(df: DataFrame,
                  indexName: String,
                  extraParams: Map[String, String] = Map()): Unit = {
    AzureSearchWriter.write(df,
      Map("subscriptionKey" -> azureSearchKey,
        "actionCol" -> "searchAction",
        "serviceName" -> testServiceName,
        "indexJson" ->  createMetIndexTest(indexName)))
//        createSimpleIndexJson(indexName))
  }

  def assertSize(indexName: String, size: Int): Unit = {
    assert(SearchIndex.getStatistics(indexName, azureSearchKey, testServiceName)._1 == size)
    ()
  }

  test("Run azure-search tests with waits") {
    val testsToRun = Set(1, 2)//, 3)

    def dependsOn(testNumber: Int, f: => Unit): Unit = {
      if (testsToRun(testNumber)) {
        println(s"Running code for test $testNumber")
        f
      }
    }

    //create new index and add docs
    lazy val in1 = generateIndexName()
    dependsOn(1, writeHelper(df4, in1))

    //push docs to existing index
    lazy val in2 = generateIndexName()
    lazy val dfA = df10.limit(4)
    lazy val dfB = df10.except(dfA)
    dependsOn(2, writeHelper(dfA, in2))

    dependsOn(2, retryWithBackoff({
      if (getExisting(azureSearchKey, testServiceName).contains(in2)) {
        writeHelper(dfB, in2)
      } else {
        throw new RuntimeException("No existing service found")
      }
    }))

    //push docs with custom batch size
    lazy val in3 = generateIndexName()
    dependsOn(3, writeHelper(bigDF, in3, Map("batchSize" -> "2000")))

    dependsOn(1, retryWithBackoff(assertSize(in1, 4)))
    dependsOn(2, retryWithBackoff(assertSize(in2, 10)))
    dependsOn(3, retryWithBackoff(assertSize(in3, 10000)))

  }

  test("Throw useful error when given badly formatted json") {
    val in = generateIndexName()
    val badJson =
      s"""
         |{
         |    "name": "$in",
         |    "fields": [
         |      {
         |        "name": "id",
         |        "type": "Edm.String",
         |        "key": true,
         |        "facetable": false
         |      },
         |    {
         |      "name": "someCollection",
         |      "type": "Collection(Edm.String)",
         |      "searchable": false,
         |      "sortable": true,
         |      "facetable": false
         |    },
         |    {
         |      "name": "text",
         |      "type": "Edm.String",
         |      "filterable": false,
         |      "sortable": false,
         |      "facetable": false
         |    }
         |    ]
         |  }
    """.stripMargin

    assertThrows[IllegalArgumentException] {
      SearchIndex.createIfNoneExists(azureSearchKey, testServiceName, badJson)
    }

  }

  test("Throw useful error when given mismatched schema and document fields") {
    val mismatchDF = (0 until 4)
      .map { i => ("upload", s"$i", s"file$i", s"text$i") }
      .toDF("searchAction", "badkeyname", "fileName", "text")
    assertThrows[IllegalArgumentException] {
      writeHelper(mismatchDF, generateIndexName())
    }
  }

  test("Write Met data to a search index"){
    // "/vagrant_data/metartworks_with_neighbors_cleaned.csv"
    //Create a SparkContext to initialize Spark

    //val df = spark.read.option("header", true).csv("/vagrant_data/metartworks_with_neighbors_cleaned_v3.csv")

    val df_in = session.read
              .format("csv")
              .option("header", true)
              //.option("delimiter", "^")
              .load("/vagrant_data/metartworks_with_neighbors_cleaned_v2.csv")
    val df = df_in.na.drop() //fill("''")
    val aug_df = df.withColumn("searchAction", lit("upload"))
    val test_df = aug_df.select("ObjectID", "Department", "Title", "Culture", "Medium", "Classification", "LinkResource", "PrimaryImageUrl", "Neighbors", "searchAction").limit(1)
    lazy val in1 = generateIndexName()
    println(in1)
    //test_df.printSchema()
    //test_df.show(5, false)
    writeHelper(test_df, in1)
//    println(SearchIndex.getStatistics("test--330914609", azureSearchKey, testServiceName)._1)
  }



}
